package mlxrunner

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/mlxrunner/kvsize"
)

// Client wraps an MLX runner subprocess to implement llm.LlamaServer for LLM models.
type Client struct {
	port      int
	modelName string
	// contextLength is what the runner reports; softContextLength is the
	// recommended limit to avoid poor performance. Atomic because Load may
	// clamp the soft limit down to what VRAM can hold while Ping reads it.
	contextLength     atomic.Int64
	softContextLength atomic.Int64
	// numCtxAuto records that num_ctx came from Ollama's VRAM-tier default
	// rather than from the request, the model or the environment. An automatic
	// rung is clamped to fit; an explicit one is refused. See admit.
	numCtxAuto bool
	memory     atomic.Uint64
	// kvEstimate prices the model's per-layer caches at a context rung. Set
	// from the manifest's config.json in NewClient, nil when that could not be
	// read; tests substitute their own.
	kvEstimate func(numCtx int) kvsize.Estimate
	done       chan struct{}
	doneErr    error // valid after done is closed
	client     *http.Client
	status     *llm.StatusWriter
	mu         sync.Mutex
	cmd        *exec.Cmd
}

// NewClient prepares a new MLX runner client for LLM models.
// The subprocess is not started until Load() is called.
//
// numCtx is the context rung the request resolved to, and numCtxAuto reports
// whether it came from Ollama's automatic VRAM-tier default. Load needs both:
// the rung to price the KV cache, and its provenance to decide between clamping
// and refusing when it does not fit.
func NewClient(modelName string, numCtx int, numCtxAuto bool) (*Client, error) {
	if err := checkPlatformSupport(); err != nil {
		return nil, err
	}

	c := &Client{
		modelName:  modelName,
		numCtxAuto: numCtxAuto,
		done:       make(chan struct{}),
		client:     http.DefaultClient,
	}
	c.softContextLength.Store(int64(numCtx))

	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}
	c.memory.Store(uint64(modelManifest.TotalTensorSize()))

	// The geometry a KV estimate needs is in config.json, which the manifest
	// already carries; no weights are loaded to price a rung.
	config, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		// Not fatal: admission falls back to weights only, which is what it
		// did before the estimator existed.
		slog.Warn("MLX admission cannot read config.json; pricing weights only",
			"model", modelName, "error", err)
		return c, nil
	}
	// draft/config.json is the default location a draft model's config lands
	// at (x/mlxrunner/model/root.go readDraftConfig); it is absent for most
	// models and a manifest may point elsewhere, in which case the draft's
	// caches go unpriced, which under-prices rather than over-refuses.
	draft, _ := modelManifest.ReadConfig("draft/config.json")
	c.kvEstimate = func(numCtx int) kvsize.Estimate { return kvsize.Model(config, draft, numCtx) }

	return c, nil
}

func checkPlatformSupport() error {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH != "arm64" {
			return fmt.Errorf("MLX on macOS requires Apple Silicon (arm64), got %s", runtime.GOARCH)
		}
		return nil
	case "linux", "windows":
		return nil
	default:
		return fmt.Errorf("MLX is not supported on %s", runtime.GOOS)
	}
}

// WaitUntilRunning waits for the subprocess to be ready.
func (c *Client) WaitUntilRunning(ctx context.Context) error {
	timeout := time.After(envconfig.LoadTimeout())
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.done:
			if msg := c.status.LastError(); msg != "" {
				return fmt.Errorf("mlx runner failed: %s (exit: %v)", msg, c.doneErr)
			}
			return fmt.Errorf("mlx runner exited unexpectedly: %w", c.doneErr)
		case <-timeout:
			if msg := c.status.LastError(); msg != "" {
				return fmt.Errorf("timeout waiting for mlx runner: %s", msg)
			}
			return errors.New("timeout waiting for mlx runner to start")
		case <-ticker.C:
			if err := c.Ping(ctx); err == nil {
				slog.Info("mlx runner is ready", "port", c.port)
				return nil
			}
		}
	}
}

type CompletionRequest struct {
	Prompt string
	// Format carries the request's structured-output constraint to the
	// runner (ADR 0009); the handler compiles it at admission.
	Format                     json.RawMessage
	Media                      []llm.MediaData
	Options                    api.Options
	Logprobs                   bool
	TopLogprobs                int
	IncludeIntermediateMetrics bool
}

type CompletionResponse struct {
	Content    string
	Done       bool
	DoneReason int

	PromptEvalCount       int
	PromptEvalCachedCount *int
	PromptEvalDuration    time.Duration
	EvalCount             int
	EvalDuration          time.Duration

	Logprobs []llm.Logprob

	Error *api.StatusError
}

// Close terminates the subprocess.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil && c.cmd.Process != nil {
		slog.Info("stopping mlx runner subprocess", "pid", c.cmd.Process.Pid)
		c.cmd.Process.Signal(os.Interrupt)

		select {
		case <-c.done:
		case <-time.After(5 * time.Second):
			c.cmd.Process.Kill()
		}
		c.cmd = nil
	}
	return nil
}

// Completion implements llm.LlamaServer.
func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	creq := CompletionRequest{
		Prompt:                     req.Prompt,
		Format:                     req.Format,
		Media:                      req.Media,
		Logprobs:                   req.Logprobs,
		TopLogprobs:                req.TopLogprobs,
		IncludeIntermediateMetrics: req.IncludeIntermediateMetrics,
	}
	if req.Options != nil {
		creq.Options = *req.Options
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}

	httpURL := fmt.Sprintf("http://127.0.0.1:%d/completion", c.port)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", httpURL, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		if errMsg := c.status.LastError(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: strings.TrimSpace(string(respBody))}
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		// The caller may cancel from inside fn (the structured-outputs
		// transition does); chunks already buffered must not keep
		// flowing after that.
		if err := ctx.Err(); err != nil {
			return err
		}

		var raw CompletionResponse
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			slog.Debug("mlx response parse error", "error", err, "line", string(scanner.Bytes()))
			continue
		}

		if raw.Error != nil {
			return *raw.Error
		}

		cresp := llm.CompletionResponse{
			Content:               raw.Content,
			Done:                  raw.Done,
			DoneReason:            llm.DoneReason(raw.DoneReason),
			PromptEvalCount:       raw.PromptEvalCount,
			PromptEvalCachedCount: raw.PromptEvalCachedCount,
			PromptEvalDuration:    raw.PromptEvalDuration,
			EvalCount:             raw.EvalCount,
			EvalDuration:          raw.EvalDuration,
			Logprobs:              raw.Logprobs,
		}

		fn(cresp)
		if cresp.Done {
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		if errMsg := c.status.LastError(); errMsg != "" {
			return fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return err
	}
	return nil
}

func (c *Client) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	return errors.New("MLX runner does not support native llama-server chat")
}

func (c *Client) ApplyChatTemplate(ctx context.Context, req llm.ChatRequest) (string, error) {
	return "", errors.New("MLX runner does not support native llama-server chat templates")
}

func (c *Client) ContextLength() int {
	return int(c.contextLength.Load())
}

func (c *Client) reportedContextLength(modelContextLength int) int {
	soft := int(c.softContextLength.Load())
	if soft > 0 && (modelContextLength == 0 || soft < modelContextLength) {
		return soft
	}
	return modelContextLength
}

// Detokenize implements llm.LlamaServer.
func (c *Client) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("not supported")
}

// Embedding implements llm.LlamaServer.
func (c *Client) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	return nil, 0, errors.New("not supported")
}

// GetDeviceInfos implements llm.LlamaServer.
func (c *Client) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return nil
}

// GetPort implements llm.LlamaServer.
func (c *Client) GetPort() int {
	return c.port
}

// HasExited implements llm.LlamaServer.
func (c *Client) HasExited() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

// autoContextFloor is the lowest rung the automatic-num_ctx clamp will step
// down to. Below it a model is not usefully servable anyway, and refusing is
// more honest than serving a window nothing fits in.
const autoContextFloor = 2048

// admissionHeadroom is what a load needs beyond weights and KV: prefill
// activations, the vision tower on a multi-image request, and the transient
// double-hold when a cache grows by Concatenate.
//
// PLACEHOLDER, to be calibrated in the GPU phase against the runner's own
// `peak memory` line. It is biased deliberately LOW: over-refusal on a serving
// host is worse than the over-admission we have today, and the operator's
// OLLAMA_MLX_MEMORY_LIMIT remains the hard ceiling. The measured gap it stands
// in for is large -- weights 17-24 GB against peaks of 24-36 GiB
// (docs/maxusai/mlx-admission-prices-weights-only.md) -- so this is a first
// margin, not a model of it.
func admissionHeadroom(weightsPlusKV uint64) uint64 {
	const floor = 512 << 20 // 512 MiB
	if fraction := weightsPlusKV / 20; fraction > floor {
		return fraction
	}
	return floor
}

// estimate prices the model's caches at numCtx, or reports an unknown estimate
// when no config was readable at construction.
func (c *Client) estimate(numCtx int) kvsize.Estimate {
	if c.kvEstimate == nil {
		return kvsize.Estimate{NumCtx: numCtx}
	}
	return c.kvEstimate(numCtx)
}

// needFor is what a load at numCtx is expected to occupy.
func (c *Client) needFor(weights uint64, numCtx int) (need, kv, headroom uint64, est kvsize.Estimate) {
	est = c.estimate(numCtx)
	if !est.Known {
		return weights, 0, 0, est
	}
	kv = est.Total()
	headroom = admissionHeadroom(weights + kv)
	return weights + kv + headroom, kv, headroom, est
}

// admit decides whether this model may load on the given devices and returns
// the byte budget to hand the runner subprocess.
//
// It prices weights + KV(num_ctx) + headroom, not weights alone. Before this,
// 8192 / 16384 / 32768 / 65536 all admitted identically on MLX and a rung that
// could not fit was accepted and then killed mid-prefill by cudaMallocAsync
// (docs/maxusai/mlx-admission-prices-weights-only.md).
func (c *Client) admit(gpus []ml.DeviceInfo, requireFull bool) (uint64, error) {
	if len(gpus) == 0 {
		return 0, nil
	}

	weights := c.memory.Load()
	// We currently only use the first GPU with MLX
	available := gpus[0].FreeMemory
	overhead := gpus[0].MinimumMemory() + envconfig.GpuOverhead()
	if available > overhead {
		available -= overhead
	} else {
		available = 0
	}

	// Budget handed to the runner so MLX caps its allocator at what is actually
	// free. MLX's own CUDA default comes from TOTAL device memory and therefore
	// overcommits whenever anything else is resident on the card.
	vramBudget := available
	if capped, overridden := budgetWithOverride(available, os.Getenv(MemoryLimitEnv)); overridden {
		vramBudget = capped
		slog.Info("MLX memory limit overridden from the environment",
			"requested", os.Getenv(MemoryLimitEnv),
			"derived", format.HumanBytes2(available),
			"using", format.HumanBytes2(vramBudget))
	}

	numCtx := int(c.softContextLength.Load())
	need, kv, headroom, est := c.needFor(weights, numCtx)
	if !est.Known {
		// No cache rule for this architecture. Fall back to exactly what
		// admission did before the estimator existed rather than guessing: a
		// wrong estimate refuses loads that would have served.
		slog.Warn("MLX admission is pricing weights only: no KV rule for this architecture",
			"architecture", est.Arch, "model", c.modelName, "num_ctx", numCtx)
	}

	// An automatic rung was never asked for: it is Ollama's VRAM-tier default
	// (262144 on the CUDA host), so pricing it and refusing would refuse
	// everything. Step it down until it fits and serve the smaller window --
	// the same shape as reduceAutoNumCtxForLoadOOM on the llama.cpp path,
	// except done before the load rather than after an OOM.
	if est.Known && c.numCtxAuto && need > vramBudget {
		fitted, ok := c.clampAutoContext(weights, vramBudget, numCtx)
		fittedNeed, fittedKV, fittedHeadroom, _ := c.needFor(weights, fitted)
		switch {
		case ok:
			slog.Info("MLX context clamped to fit VRAM",
				"model", c.modelName, "requested", numCtx, "using", fitted,
				"weights", format.HumanBytes2(weights),
				"kv", format.HumanBytes2(fittedKV),
				"headroom", format.HumanBytes2(fittedHeadroom),
				"budget", format.HumanBytes2(vramBudget))
			c.softContextLength.Store(int64(fitted))
			numCtx, need, kv, headroom = fitted, fittedNeed, fittedKV, fittedHeadroom
		default:
			// Even the floor does not fit. An automatic rung must not turn into
			// a refusal the user cannot act on, so admit on the weights alone
			// (today's behaviour) and let the runner's own ceiling bound it.
			slog.Warn("MLX auto context does not fit even at the floor; admitting on weights alone",
				"model", c.modelName, "requested", numCtx, "floor", autoContextFloor,
				"need", format.HumanBytes2(fittedNeed),
				"budget", format.HumanBytes2(vramBudget))
			c.softContextLength.Store(int64(autoContextFloor))
			numCtx, need, kv, headroom = autoContextFloor, weights, fittedKV, fittedHeadroom
		}
	}

	// PHYSICAL shortfall: the card cannot hold the model. Evictable, so it
	// stays ErrLoadRequiredFull under requireFull -- freeing another runner
	// raises FreeMemory and a retry can succeed.
	if need > available {
		if requireFull {
			return 0, llm.ErrLoadRequiredFull
		}
		if !est.Known {
			return 0, fmt.Errorf("model requires %s but only %s are available (after %s overhead)", format.HumanBytes2(need), format.HumanBytes2(available), format.HumanBytes2(overhead))
		}
		return 0, fmt.Errorf("model requires %s (weights %s + KV cache %s at num_ctx %d + %s headroom) but only %s are available (after %s overhead); lower num_ctx or free VRAM on the device",
			format.HumanBytes2(need), format.HumanBytes2(weights), format.HumanBytes2(kv), numCtx,
			format.HumanBytes2(headroom), format.HumanBytes2(available), format.HumanBytes2(overhead))
	}

	// OPERATOR shortfall: it fits the card but not the ceiling the operator
	// set. This check has to exist separately, and it must NOT be
	// ErrLoadRequiredFull.
	//
	// Separately, because the check above reads `available` while the value
	// actually handed to the runner is `vramBudget`. Before the override
	// existed the two were the same number, so one check covered both; now
	// a cap below the model size is admitted and passed straight through to
	// the subprocess. Measured: model 18.29 GiB, 88 GiB free,
	// OLLAMA_MLX_MEMORY_LIMIT=8589934592 -> admitted even with
	// requireFull=true, ceiling 10.3 GiB under the weights. Nothing
	// downstream catches it either: runner.go evaluates the weights before
	// configureMemoryLimit runs, so the cap lands on an already-resident
	// model, and that function only guards upward.
	//
	// And NOT ErrLoadRequiredFull, because sched.go turns that into "evict a
	// runner and retry". Eviction raises FreeMemory; the operator's cap is a
	// constant that free memory cannot move, so every retry would recompute
	// the same budget and fail identically -- evicting every other model to
	// satisfy a constraint eviction cannot satisfy. A plain error naming the
	// variable is the only correct answer, and it tells the operator which
	// knob to turn.
	if need > vramBudget {
		return 0, fmt.Errorf("model requires %s but %s caps the MLX budget at %s", format.HumanBytes2(need), MemoryLimitEnv, format.HumanBytes2(vramBudget))
	}

	// The one line the GPU phase compares against the runner's `peak memory`.
	slog.Info("MLX admission priced the context rung",
		"model", c.modelName,
		"architecture", est.Arch,
		"num_ctx", numCtx,
		"num_ctx_auto", c.numCtxAuto,
		"weights", format.HumanBytes2(weights),
		"kv", format.HumanBytes2(kv),
		"headroom", format.HumanBytes2(headroom),
		"need", format.HumanBytes2(need),
		"available", format.HumanBytes2(available),
		"budget", format.HumanBytes2(vramBudget),
		"layers", est.Layers.Total(),
		"layers_full", est.Layers.Attention,
		"layers_sliding", est.Layers.Sliding,
		"layers_recurrent", est.Layers.Recurrent)

	return vramBudget, nil
}

// clampAutoContext halves the rung until the load fits the budget, stopping at
// autoContextFloor. It reports the rung and whether one was found that fits.
func (c *Client) clampAutoContext(weights, budget uint64, numCtx int) (int, bool) {
	rung := numCtx
	for rung > autoContextFloor {
		rung = max(rung/2, autoContextFloor)
		if need, _, _, _ := c.needFor(weights, rung); need <= budget {
			return rung, true
		}
	}
	return rung, false
}

// Load checks whether the model fits in GPU memory and starts the subprocess.
func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	vramBudget, err := c.admit(gpus, requireFull)
	if err != nil {
		return nil, err
	}

	// Find a free port
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		if l, err := net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152
	}
	c.port = port

	// Get the current executable path
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("unable to lookup executable path: %w", err)
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	// Spawn subprocess: ollama runner --mlx-engine --model <name> --port <port>
	cmd := exec.Command(exe, "runner", "--mlx-engine", "--model", c.modelName, "--port", strconv.Itoa(port))
	cmd.Env = os.Environ()

	// Set library path environment variable for MLX libraries
	// Linux: LD_LIBRARY_PATH, Windows: PATH
	var libPathEnvVar string
	switch runtime.GOOS {
	case "linux":
		libPathEnvVar = "LD_LIBRARY_PATH"
	case "windows":
		libPathEnvVar = "PATH"
	}

	if libPathEnvVar != "" {
		libraryPaths := []string{ml.LibOllamaPath}
		if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_*")); err == nil {
			libraryPaths = append(libraryPaths, mlxDirs...)
		}

		if existingPath, ok := os.LookupEnv(libPathEnvVar); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(existingPath)...)
		}

		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		found := false
		for i := range cmd.Env {
			envName := cmd.Env[i]
			if runtime.GOOS == "windows" {
				envName = strings.ToUpper(envName)
			}
			if strings.HasPrefix(envName, libPathEnvVar+"=") {
				cmd.Env[i] = libPathEnvVar + "=" + pathEnvVal
				found = true
				break
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, libPathEnvVar+"="+pathEnvVal)
		}
		slog.Debug("mlx subprocess library path", libPathEnvVar, pathEnvVal)
	}

	// Point MLX's JIT compiler at our bundled CUDA runtime headers.
	// MLX resolves headers via $CUDA_PATH/include/*.h (and checks CUDA_HOME first).
	// Always use bundled headers to avoid version mismatches with any
	// system-installed CUDA toolkit.
	if mlxDirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "mlx_cuda_*")); err == nil {
		for _, d := range mlxDirs {
			if _, err := os.Stat(filepath.Join(d, "include")); err == nil {
				setEnv(cmd, "CUDA_PATH", d)
				setEnv(cmd, "CUDA_HOME", d)
				slog.Debug("mlx subprocess CUDA headers", "CUDA_PATH", d)
				break
			}
		}
	}

	mlxRunnerEnvDefaults(cmd)

	c.cmd = cmd

	status := llm.NewStatusWriter(os.Stderr)
	c.status = status
	// os/exec serializes Write calls when shared, which keeps the status writer
	// from seeing concurrent stdout/stderr fragments.
	cmd.Stdout = status
	cmd.Stderr = status

	if vramBudget > 0 {
		setEnv(cmd, MemoryLimitEnv, strconv.FormatUint(vramBudget, 10))
		slog.Debug("mlx subprocess memory budget", MemoryLimitEnv, format.HumanBytes2(vramBudget))
	}

	slog.Info("starting mlx runner subprocess", "model", c.modelName, "port", c.port)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx runner: %w", err)
	}

	// Reap subprocess when it exits
	go func() {
		c.doneErr = cmd.Wait()
		close(c.done)
	}()

	return nil, nil
}

// ModelPath implements llm.LlamaServer.
func (c *Client) ModelPath() string {
	return c.modelName
}

// Pid implements llm.LlamaServer.
func (c *Client) Pid() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cmd != nil && c.cmd.Process != nil {
		return c.cmd.Process.Pid
	}
	return -1
}

type statusResponse struct {
	Status        int
	Progress      int
	ContextLength int
	Memory        uint64
}

// Ping implements llm.LlamaServer.
func (c *Client) Ping(ctx context.Context) error {
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/status", c.port)
	req, err := http.NewRequestWithContext(ctx, "GET", reqURL, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: %d", resp.StatusCode)
	}

	var status statusResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return err
	}

	c.contextLength.Store(int64(c.reportedContextLength(status.ContextLength)))
	c.memory.Store(status.Memory)

	return nil
}

// Tokenize implements llm.LlamaServer.
func (c *Client) Tokenize(ctx context.Context, content string) ([]int, error) {
	reqURL := fmt.Sprintf("http://127.0.0.1:%d/v1/tokenize", c.port)
	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, strings.NewReader(content))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "text/plain")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var tokens []int
	if err := json.NewDecoder(resp.Body).Decode(&tokens); err != nil {
		return nil, err
	}

	return tokens, nil
}

func (c *Client) currentMemory() uint64 {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	c.Ping(ctx) //nolint:errcheck
	return c.memory.Load()
}

// MemorySize implements llm.LlamaServer.
func (c *Client) MemorySize() (total, vram uint64) {
	mem := c.currentMemory()
	return mem, mem
}

// VRAMByGPU implements llm.LlamaServer.
func (c *Client) VRAMByGPU(id ml.DeviceID) uint64 {
	return c.currentMemory()
}

var _ llm.LlamaServer = (*Client)(nil)

// CacheThrashingCheckEnv is MLX's own switch for the CUDA graph-cache
// "thrashing check" (ml-explore/mlx #2600). The fork turns it off for the
// runner unless the operator sets it; see mlxRunnerEnvDefaults.
const CacheThrashingCheckEnv = "MLX_ENABLE_CACHE_THRASHING_CHECK"

// mlxRunnerEnvDefaults applies the fork's defaults for MLX's own knobs to the
// runner environment. A non-empty value the operator set on the server is left
// alone, so the advisory below can be re-enabled by exporting the variable.
//
// WHY THE THRASHING CHECK IS OFF. MLX's CUDA backend keys a graph cache by
// shape, and the check is a LIFETIME miss counter that throws once misses pass
// 2 x MLX_CUDA_GRAPH_CACHE_SIZE (default 400). A throw out of graph commit is
// not something this runner can catch and continue from: the request dies,
// the deferred prefix-cache close then fails on the encoder the first throw
// left behind, and the log blames cudaGraphAddDependencies. A think-on decode
// reaches the threshold in ~700 distinct prefill lengths, on every image we
// ship; with the check off the LRU still evicts and nothing measurable
// changes (120/120 and 400/400 clean under the conditions that failed
// 120/120). It is a performance advisory, and an advisory that kills the
// request is the wrong default for a server. Measurements and mechanism:
// docs/maxusai/mlx-thrash-check-masks-as-cudagraph.md.
func mlxRunnerEnvDefaults(cmd *exec.Cmd) {
	if os.Getenv(CacheThrashingCheckEnv) == "" {
		setEnv(cmd, CacheThrashingCheckEnv, "0")
	}
}

// setEnv sets or replaces an environment variable in cmd.Env.
func setEnv(cmd *exec.Cmd, key, value string) {
	entry := key + "=" + value
	prefix := strings.ToUpper(key + "=")
	for i, e := range cmd.Env {
		if strings.HasPrefix(strings.ToUpper(e), prefix) {
			cmd.Env[i] = entry
			return
		}
	}
	cmd.Env = append(cmd.Env, entry)
}

// budgetWithOverride applies an operator-set OLLAMA_MLX_MEMORY_LIMIT on top of
// the budget derived from free device memory, returning the value to use and
// whether an override was applied.
//
// WHY AN OVERRIDE EXISTS AT ALL. setEnv REPLACES whatever the environment
// holds, so a limit set on the container was silently overwritten by the
// derivation and the knob did nothing. That is not merely inconvenient: MLX's
// allocator grows toward whatever ceiling it is given, so the ceiling is the
// main lever on steady-state footprint — and on a shared card an 18 GB model
// handed 82 GiB is a bad neighbour. It was also untestable: a sweep over
// 24/48/82 GiB produced three identical runs, all at 82, because nothing in the
// output revealed that the request had been discarded.
//
// AN OVERRIDE MAY ONLY ASK FOR LESS. Clamping to the derived value is what
// keeps this from reintroducing the overcommit that motivated deriving from
// FREE memory rather than MLX's own default, which comes from TOTAL device
// memory and overcommits whenever anything else is resident.
func budgetWithOverride(available uint64, env string) (uint64, bool) {
	v, ok := parseMemoryBudget(env)
	if !ok || v <= 0 {
		return available, false
	}
	if uint64(v) >= available {
		// Reported as an override so the log records that a request was seen
		// and what it resolved to; silently ignoring it is the behaviour this
		// function exists to remove.
		return available, true
	}
	return uint64(v), true
}
