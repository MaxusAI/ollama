package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/structured"
	"github.com/ollama/ollama/x/tokenizer"
)

// Request is a short-lived struct that carries a completion request through
// a channel from the HTTP handler to the runner goroutine. The ctx field
// must travel with the request so that cancellation propagates across the
// channel boundary.
type Request struct {
	CompletionRequest
	Responses chan CompletionResponse
	Pipeline  func(context.Context, Request) error

	Ctx         context.Context //nolint:containedctx // Queued requests carry caller cancellation to the runner.
	Tokens      []int32
	MediaItems  []mediaItem
	Layout      any // opaque PrepareMedia layout state, stamped on every batch
	SamplerOpts sample.Options

	// Constraint is the compiled format grammar, nil when the request
	// carries no format. Populated by Prepare.
	Constraint *structured.Grammar
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	Sampler       *sample.Sampler
	cache         *prefixCache
	contextLength int
	mlxThread     *mlxthread.Thread
	// spec is the speculative-decoding subsystem. Nil when the model ships no
	// draft head.
	spec *speculation

	// Constrained-sampling vocabulary index, built lazily on the first
	// request that carries a format.
	constraintOnce   sync.Once
	constraintVocab  *structured.Vocab
	constraintPieces [][]byte
}

func (r *Runner) Load(modelName string) error {
	root, err := model.Open(modelName)
	if err != nil {
		return err
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return err
	}

	// Load all tensor blobs from manifest
	tensors, err := loadTensorsFromManifest(root)
	if err != nil {
		return err
	}

	// Assign weights to model (model-specific logic). Target and draft weights
	// must be loaded before sweeping so tensors from a combined manifest are
	// not discarded before the draft model can retain them.
	if err := m.LoadWeights(tensors); err != nil {
		return err
	}

	var draftModel base.DraftModel
	draft, err := base.NewDraft(root, m)
	if err != nil {
		return err
	}
	if draft != nil {
		if err := draft.LoadWeights(tensors); err != nil {
			return err
		}
		draftModel = draft
	} else if sd, ok := m.(base.SelfDraft); ok {
		// Inline draft head: already loaded with the target; nil if none shipped.
		draftModel = sd.SelfDraft()
	}

	collected := mlx.Collect(m)
	if draft != nil {
		draftArrays := mlx.Collect(draft)
		collected = append(collected, draftArrays...)
		if root.Draft != nil {
			slog.Info("Loaded draft model", "tensor_prefix", root.Draft.TensorPrefix, "config", root.Draft.Config, "arrays", len(draftArrays))
		} else {
			slog.Info("Loaded draft model", "arrays", len(draftArrays))
		}
	}
	for _, arr := range collected {
		mlx.Pin(arr)
	}
	mlx.Sweep()
	mlx.Eval(collected...)
	configureWiredMemory()
	// After the weights are resident, so the "previous" it logs and any cache
	// it trims reflect a loaded model rather than an empty allocator.
	configureCacheLimit()

	r.Model = m
	r.Tokenizer = m.Tokenizer()
	r.contextLength = m.MaxContextLength()
	caches := m.NewCaches()
	draftCaches := newDraftCaches(draftModel)
	r.cache = newPrefixCache(slices.Concat(caches, draftCaches))
	r.Sampler = sample.New(r.contextLength)
	r.spec = newSpeculation(r, draftModel, caches, draftCaches)

	mlx.EnableCompile()

	return nil
}

// newDraftCaches returns nil when the model ships no draft.
func newDraftCaches(draft base.DraftModel) []cache.Cache {
	if draft == nil {
		return nil
	}
	return draft.NewCaches()
}

// MemoryLimitEnv carries the parent's view of FREE device memory, in bytes, to
// the runner subprocess. Set by the MLX client, which computes it from
// ml.DeviceInfo.FreeMemory less the per-device minimum and OLLAMA_GPU_OVERHEAD.
const MemoryLimitEnv = "OLLAMA_MLX_MEMORY_LIMIT"

// configureMemoryLimit caps the allocator on backends with no wired-residency
// concept — everything except Metal.
//
// MLX is not uncapped on CUDA: it defaults the limit to a fraction of TOTAL
// device memory (measured 90.22 GiB on a 95.6 GiB card). That is the wrong
// denominator on a shared GPU. With other processes already holding tens of
// gigabytes, MLX still believes the whole card is its own, so a growing KV
// cache allocates past what is actually free and cudaMallocAsync fails. MLX
// aborts the process on a failed allocation rather than returning an error, so
// the runner dies and the request 500s.
//
// The parent already computes the right number and previously discarded it
// after an admission check. Prefer it; fall back to reporting MLX's own default
// so the ceiling is visible rather than assumed.
func configureMemoryLimit(active int, wsErr error) {
	limit, err := mlx.MemoryLimit()
	if err != nil {
		slog.Warn("Unable to query MLX recommended working set; using pageable memory",
			"error", wsErr, "memory_limit_error", err)
		return
	}

	budget, ok := parseMemoryBudget(os.Getenv(MemoryLimitEnv))
	if !ok {
		slog.Warn("Unable to query MLX recommended working set (Metal-only device key); "+
			"relying on the backend default, which is derived from TOTAL device memory",
			"error", wsErr,
			"active", mlx.PrettyBytes(active),
			"backend_limit", mlx.PrettyBytes(limit))
		return
	}

	// Never raise the backend's own ceiling — only lower it to what is free.
	if budget >= limit {
		slog.Debug("MLX memory budget is not below the backend limit; leaving it alone",
			"budget", mlx.PrettyBytes(budget), "backend_limit", mlx.PrettyBytes(limit))
		return
	}

	previous, err := mlx.SetMemoryLimit(budget)
	if err != nil {
		slog.Warn("Unable to apply MLX memory limit; keeping the backend default",
			"budget", mlx.PrettyBytes(budget), "error", err)
		return
	}
	slog.Info("Configured MLX memory limit from free device memory",
		"active", mlx.PrettyBytes(active),
		"limit", mlx.PrettyBytes(budget),
		"previous", mlx.PrettyBytes(previous))
}

// CacheLimitEnv bounds MLX's RETAINED buffer cache, in bytes. Unset leaves
// MLX's own limit in place; 0 means retain nothing. See configureCacheLimit for
// the measured trade and why there is no default.
const CacheLimitEnv = "OLLAMA_MLX_CACHE_LIMIT"

// configureCacheLimit bounds what MLX keeps after it is finished with it.
//
// The memory limit above caps TOTAL allocation and decides when an allocation
// FAILS. It does not make MLX give anything back, and MLX's allocator retains
// freed blocks for reuse. Measured on gemma4:31b-nvfp4, one 3072x1728 image:
// active 18.29 GiB against cache 13.16 GiB, for a 32.36 GiB process where
// llama.cpp does the identical work — same image, same num_ctx, same
// prompt_eval_count — in 23.03 GiB. The cache IS the difference.
//
// That matters beyond tidiness on a shared card: a large retained cache is a
// candidate cause of an out-of-memory abort seen on qwen3.6:35b-a3b-nvfp4 at
// the same geometry while nvidia-smi reported 85 GiB free, because the pool
// holds memory the driver can no longer hand to a big contiguous request.
//
// NO DEFAULT, ON MEASUREMENT. A bounded default was shipped and then withdrawn:
// the trade is sharp and has no comfortable middle. Measured n=3 on
// gemma4:31b-nvfp4, one 3072x1728 image, decode tok/s against peak footprint:
//
//	  4 GiB   29.44 tok/s   28,749 MiB    <- all of the footprint win
//	  8 GiB   34.73 tok/s   32,779 MiB
//	 16 GiB   34.77 tok/s   33,295 MiB
//	 90 GiB   34.98 tok/s   33,276 MiB    <- MLX's own default
//
// Throughput recovers fully by 8 GiB, but 8 GiB saves 497 MiB -- nothing. The
// entire 4.5 GB saving sits at 4 GiB, and 4 GiB costs 15.8% decode, because the
// transient working set is ~7 GiB (peak 25.37 against active 18.29) and a
// smaller cache makes the allocator round-trip to the driver inside every
// forward.
//
// So a default at 8 GiB would buy nothing while adding a surprise, and a default
// at 4 GiB would silently cost every user a sixth of their throughput to save
// memory most of them are not short of. Neither is worth doing on the operator's
// behalf. The knob is opt-in: set OLLAMA_MLX_CACHE_LIMIT when the footprint
// matters more than the speed -- a shared card, or a model that OOMs at large
// geometries -- and pay the cost knowingly.
func configureCacheLimit() {
	limit, ok := parseCacheLimit(os.Getenv(CacheLimitEnv))
	if !ok {
		return
	}
	previous, err := mlx.SetCacheLimit(limit)
	if err != nil {
		slog.Warn("Unable to apply MLX cache limit; keeping the backend default",
			"limit", mlx.PrettyBytes(limit), "error", err)
		return
	}
	slog.Info("Configured MLX buffer cache limit",
		"limit", mlx.PrettyBytes(limit),
		"previous", mlx.PrettyBytes(previous),
		"memory", mlx.Memory{})
}

// parseCacheLimit is parseMemoryBudget with ZERO ADMITTED. The two look alike
// and mean different things: a memory budget of 0 is meaningless (it would
// forbid every allocation), so parseMemoryBudget rejects it, but a CACHE limit
// of 0 is the most useful value in the range -- "retain nothing", the setting
// that makes MLX return every freed buffer instead of holding it.
//
// Reusing parseMemoryBudget here silently discarded OLLAMA_MLX_CACHE_LIMIT=0
// and left the backend default in place, so a measurement of that setting
// reported the DEFAULT's footprint under the cap-0 label -- 33,134 MiB,
// byte-identical to the unset arm, which is what gave it away.
func parseCacheLimit(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil || v > math.MaxInt {
		return 0, false
	}
	return int(v), true
}

func parseMemoryBudget(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil || v == 0 || v > math.MaxInt {
		return 0, false
	}
	return int(v), true
}

func configureWiredMemory() {
	if !mlx.GPUIsAvailable() {
		return
	}

	active := mlx.ActiveMemory()
	maxRecommended, err := mlx.MaxRecommendedWorkingSetSize()
	if err != nil {
		// max_recommended_working_set_size is a Metal device key, and so is the
		// wired-residency concept it feeds. Returning here left non-Metal
		// backends with NO cap at all: on CUDA a growing KV cache allocated
		// until cudaMallocAsync failed, and MLX aborts the process on a failed
		// allocation rather than returning an error, so the runner died and the
		// request 500'd. Report the allocator limit that IS portable so the
		// backend's own ceiling is visible rather than assumed.
		configureMemoryLimit(active, err)
		return
	}

	limit := min(active, maxRecommended)
	previous, err := mlx.SetWiredLimit(limit)
	if err != nil {
		slog.Warn("Unable to configure MLX wired memory; using pageable memory",
			"active", mlx.PrettyBytes(active),
			"limit", mlx.PrettyBytes(limit),
			"error", err)
		return
	}

	if active > maxRecommended {
		slog.Warn("MLX model exceeds the recommended working set; performance may be degraded",
			"active", mlx.PrettyBytes(active),
			"recommended", mlx.PrettyBytes(maxRecommended))
	}
	// Limiting residency to the loaded model's active allocations avoids
	// reserving the remaining capacity for growing KV caches.
	slog.Debug("Configured MLX wired memory",
		"active", mlx.PrettyBytes(active),
		"limit", mlx.PrettyBytes(limit),
		"previous", mlx.PrettyBytes(previous))
}

// loadTensorsFromManifest loads all tensor blobs from the manifest into a
// flat map, deduplicating by digest and remapping safetensors key suffixes.
//
// Uses a two-phase approach: first loads all raw tensors, then remaps
// .bias → _qbias with complete knowledge of which base names have .scale
// entries. This avoids a race condition where Go map iteration order could
// cause .bias to be processed before .scale within the same blob.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	// Phase 1: Load all tensors raw from all blobs
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	// Phase 2: Identify all base names that have .scale tensors and remap them
	scaleBaseNames := make(map[string]bool)
	allTensors := make(map[string]*mlx.Array, len(rawTensors))
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			allTensors[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}

	// Phase 3: Process remaining tensors with complete scale knowledge
	for name, arr := range rawTensors {
		if strings.HasSuffix(name, ".scale") {
			continue // already handled
		}
		if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				allTensors[baseName+"_qbias"] = arr
			} else {
				allTensors[name] = arr
			}
		} else {
			allTensors[name] = arr
		}
	}

	slog.Info("Loaded tensors from manifest", "count", len(allTensors))
	return allTensors, nil
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				err := r.runRequest(request)
				if err != nil {
					slog.Info("Request terminated", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
				}

				close(request.Responses)

				// Report first, stop second. The caller gets a StatusError
				// describing the failure; only then does the runner exit, so
				// the scheduler reloads a clean one rather than this process
				// continuing on MLX state a failed evaluation abandoned.
				var fatal fatalRunnerError
				if errors.As(err, &fatal) {
					return err
				}
			}
		}
	})

	srv := &http.Server{Addr: net.JoinHostPort(host, port), Handler: mux}

	// Without this the worker returning an error cancels ctx but ListenAndServe
	// keeps running, so g.Wait() never returns and the "stop second" above
	// never takes effect.
	g.Go(func() error {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	})

	g.Go(func() error {
		slog.Info("Starting HTTP server", "host", host, "port", port)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	})

	return g.Wait()
}

// fatalRunnerError marks a failure the runner must not continue past. MLX
// treats a failed graph evaluation as unrecoverable and mlxthread re-panics it
// onto this goroutine deliberately; recovering lets the caller be told what
// happened instead of reading a stack from a dead subprocess, but the runner
// still stops afterwards rather than serving on state MLX has abandoned.
type fatalRunnerError struct{ err error }

func (e fatalRunnerError) Error() string { return e.err.Error() }
func (e fatalRunnerError) Unwrap() error { return e.err }

// recoverRequest converts a panic raised while evaluating a request into an
// error, so the existing error path in Run reports it as a StatusError to the
// client. Allocation failures get the one thing a stack trace cannot give the
// operator: what to change.
func recoverRequest(err *error) {
	v := recover()
	if v == nil {
		return
	}

	msg := fmt.Sprint(v)
	wrapped := fmt.Errorf("mlx runner aborted: %s", msg)
	if strings.Contains(msg, "out of memory") {
		wrapped = fmt.Errorf("%w; the device ran out of memory during evaluation — "+
			"lower num_ctx or free VRAM on the device", wrapped)
	}
	slog.Error("Recovered a panic while evaluating a request; stopping the runner",
		"error", msg)
	*err = fatalRunnerError{err: wrapped}
}

func (r *Runner) runRequest(request Request) (err error) {
	defer recoverRequest(&err)

	if r.mlxThread == nil {
		return request.Pipeline(request.Ctx, request)
	}

	return r.mlxThread.Do(request.Ctx, func() error {
		return request.Pipeline(request.Ctx, request)
	})
}
