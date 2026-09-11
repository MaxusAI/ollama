// Command qqmm measures the three ways MLX can run an nvfp4 linear layer on
// the GPU it finds, on the layer shapes of the fork's nvfp4 models:
//
//	qmm   mlx.QuantizedMatmul: x bf16, w nvfp4. What the runner uses today
//	      (in-register dequant, bf16 tensor-core MMA; the prefill kernel is
//	      JIT-compiled, which is why the first call is reported separately).
//	qqmm  mlx.QQMM: x quantized to nvfp4 on the fly, w nvfp4. The cuBLASLt
//	      block-scaled FP4 GEMM on compute capability 10 and up, else MLX's
//	      fallback. This is the path the runner does not use.
//	bf16  mlx.Matmul: x bf16, w bf16 unquantized. cuBLASLt, the
//	      no-quantization reference.
//	dequant  mlx.Dequantize of the nvfp4 weights to bf16 on every call, then
//	      the bf16 cuBLASLt matmul: the same numbers as qmm (bf16 math on
//	      the same 4-bit weights) at the cost of a temporary bf16 copy of
//	      the layer, priced here per call.
//
// Every cell reports the wall time per call (median, p10, p90 over -iters
// after -warmup calls, plus the first call), TFLOP/s, and the error against
// an fp32 product with the unquantized weights (relative RMS and max abs),
// so qqmm's activation-quantization cost shows next to the
// weight-quantization cost the two quantized paths share.
//
// Each JSON line also carries the wall-clock window of the calls behind its
// timings (start_unix_ms, end_unix_ms), and each shape's header line in the
// text log prints the time the shape started, so a run on a shared GPU can
// be matched afterwards against another process's activity log.
//
// The tool is standalone: it needs only the MLX library the binding finds
// next to the executable (../lib/ollama/mlx_*), so run it from the ollama
// image with the binary mounted under /usr/bin:
//
//	docker run --rm --gpus all --entrypoint /usr/bin/qqmm \
//	  -v $PWD/qqmm:/usr/bin/qqmm:ro maxusai/ollama:<tag> -out /tmp/qqmm.jsonl
//
// Wall times beside other GPU work measure the contention, not the kernels.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type shape struct {
	preset, name string
	k, n         int // w is [n, k]; y[m, n] = x[m, k] · wᵀ
}

// The text-stack projections of the two dense nvfp4 models the fork serves
// (config.json of the store's manifests, 2026-09-08). MoE experts go through
// the gather path and are not measured here.
var presets = map[string][]shape{
	"gemma4-31b": {
		{"gemma4-31b", "q_proj", 5376, 8192},
		{"gemma4-31b", "o_proj", 8192, 5376},
		{"gemma4-31b", "gate_proj", 5376, 21504},
		{"gemma4-31b", "down_proj", 21504, 5376},
	},
	"qwen3.8-27b": {
		{"qwen3.8-27b", "q_proj", 5120, 6144},
		{"qwen3.8-27b", "o_proj", 6144, 5120},
		{"qwen3.8-27b", "gate_proj", 5120, 17408},
		{"qwen3.8-27b", "down_proj", 17408, 5120},
	},
}

const (
	groupSize = 16
	bits      = 4
	mode      = "nvfp4"
)

type result struct {
	Preset   string  `json:"preset"`
	Proj     string  `json:"proj"`
	M        int     `json:"m"`
	K        int     `json:"k"`
	N        int     `json:"n"`
	Method   string  `json:"method"`
	FirstMs  float64 `json:"first_ms"`
	MedianMs float64 `json:"median_ms"`
	P10Ms    float64 `json:"p10_ms"`
	P90Ms    float64 `json:"p90_ms"`
	TFLOPS   float64 `json:"tflops"`
	RelRMS   float64 `json:"rel_rms_err"`
	MaxAbs   float64 `json:"max_abs_err"`
	PeakGiB  float64 `json:"peak_gib"` // MLX's device peak so far (weights, operands, outputs, cache)
	Error    string  `json:"error,omitempty"`

	// The wall-clock window of the calls behind this row's timings, in Unix
	// milliseconds: just before the first measured call (the cold one reported
	// as first_ms) and just after the last timed one. On a shared GPU it shows
	// which rows another process's burst overlapped. Appended after the
	// original fields so their order is unchanged for existing readers.
	StartUnixMs int64 `json:"start_unix_ms"`
	EndUnixMs   int64 `json:"end_unix_ms"`
}

func main() {
	shapesFlag := flag.String("shapes", "gemma4-31b,qwen3.8-27b", "comma-separated presets, or custom name=K:N entries")
	mFlag := flag.String("m", "1,8,64,512,2048,4096", "comma-separated row counts (1 = decode, 2048 = the prefill chunk)")
	methodsFlag := flag.String("methods", "bf16,qmm,qqmm,dequant", "comma-separated subset of bf16,qmm,qqmm,dequant")
	iters := flag.Int("iters", 20, "timed calls per cell")
	warmup := flag.Int("warmup", 3, "untimed calls per cell before the timed ones (the first is reported as first_ms)")
	seed := flag.Uint64("seed", 1, "seed for the normal-distributed operands")
	out := flag.String("out", "", "append one JSON line per cell to this file")
	flag.Parse()

	// MLX-C is not thread-safe across goroutines; keep one OS thread.
	runtime.LockOSThread()
	if err := mlx.CheckInit(); err != nil {
		fmt.Fprintln(os.Stderr, "mlx:", err)
		os.Exit(1)
	}
	lib, _ := mlx.LoadedLibraryPath()
	fmt.Printf("mlx %s  cuda=%v  library=%s\n", mlx.Version(), mlx.CUDAIsAvailable(), lib)
	if !mlx.CUDAIsAvailable() && !mlx.MetalIsAvailable() {
		fmt.Fprintln(os.Stderr, "no GPU backend available")
		os.Exit(1)
	}
	mlx.SetDefaultDeviceGPU()

	shapes, err := parseShapes(*shapesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ms, err := parseInts(*mFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	methods := strings.Split(*methodsFlag, ",")

	var sink *os.File
	if *out != "" {
		sink, err = os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		defer sink.Close()
	}

	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))
	var results []result
	for _, sh := range shapes {
		fmt.Print(shapeHeader(sh, time.Now()))
		w32 := mlx.FromValues(randn(rng, sh.n*sh.k, 0.02), sh.n, sh.k)
		wbf := w32.AsType(mlx.DTypeBFloat16)
		wq, ws, wb := mlx.Quantize(wbf, groupSize, bits, mode)
		w32T := w32.Transpose(1, 0)
		wbfT := wbf.Transpose(1, 0)
		mlx.Eval(w32, wbf, wq, ws)
		mlx.Pin(w32, wbf, wq, ws, wb, w32T, wbfT)
		mlx.Sweep()
		fmt.Printf("   quantized w: %v %v, scales %v %v\n", wq.Dims(), wq.DType(), ws.Dims(), ws.DType())

		for _, m := range ms {
			x32 := mlx.FromValues(randn(rng, m*sh.k, 1.0), m, sh.k)
			xbf := x32.AsType(mlx.DTypeBFloat16)
			ref := mlx.Matmul(x32, w32T) // fp32 product with the unquantized weights
			mlx.Eval(x32, xbf, ref)
			mlx.Pin(x32, xbf, ref)
			mlx.Sweep()

			for _, method := range methods {
				var f func() *mlx.Array
				switch method {
				case "bf16":
					f = func() *mlx.Array { return mlx.Matmul(xbf, wbfT) }
				case "qmm":
					f = func() *mlx.Array {
						return mlx.QuantizedMatmul(xbf, wq, ws, wb, true, groupSize, bits, mode)
					}
				case "qqmm":
					f = func() *mlx.Array { return mlx.QQMM(xbf, wq, ws, groupSize, bits, mode, nil, nil) }
				case "dequant":
					f = func() *mlx.Array {
						w := mlx.Dequantize(wq, ws, wb, groupSize, bits, mode, nil).AsType(mlx.DTypeBFloat16)
						return mlx.Matmul(xbf, w.Transpose(1, 0))
					}
				default:
					fmt.Fprintln(os.Stderr, "unknown method", method)
					os.Exit(2)
				}
				r := result{Preset: sh.preset, Proj: sh.name, M: m, K: sh.k, N: sh.n, Method: method}
				measure(&r, f, ref, *warmup, *iters)
				r.PeakGiB = float64(mlx.PeakMemory()) / (1 << 30)
				results = append(results, r)
				printRow(r)
				if sink != nil {
					b, _ := json.Marshal(r)
					fmt.Fprintln(sink, string(b))
				}
			}
			mlx.Unpin(x32, xbf, ref)
			mlx.Sweep()
		}
		mlx.Unpin(w32, wbf, wq, ws, wb, w32T, wbfT)
		mlx.Sweep()
		mlx.ClearCache()
		fmt.Printf("   device peak so far %.2f GiB (active now %.2f GiB)\n", float64(mlx.PeakMemory())/(1<<30), float64(mlx.ActiveMemory())/(1<<30))
	}
	fmt.Printf("\ndevice peak over the run: %.2f GiB (MLX allocations; the CUDA context and JIT are on top)\n", float64(mlx.PeakMemory())/(1<<30))
	printTable(results, methods)
}

// measure times f (build the graph, evaluate, free) and fills r. A method the
// device or library rejects is recorded as an error instead of aborting.
func measure(r *result, f func() *mlx.Array, ref *mlx.Array, warmup, iters int) {
	defer func() {
		if e := recover(); e != nil {
			r.Error = fmt.Sprint(e)
			if r.EndUnixMs == 0 { // the window ends where the failing call did
				r.EndUnixMs = time.Now().UnixMilli()
			}
			mlx.Sweep()
		}
	}()
	call := func() float64 {
		t0 := time.Now()
		out := f()
		mlx.Eval(out)
		d := time.Since(t0)
		mlx.Sweep()
		return float64(d.Nanoseconds()) / 1e6
	}
	ts := timeCalls(r, call, warmup, iters, time.Now)
	sort.Float64s(ts)
	r.MedianMs = ts[len(ts)/2]
	r.P10Ms = ts[len(ts)/10]
	r.P90Ms = ts[len(ts)*9/10]
	r.TFLOPS = 2 * float64(r.M) * float64(r.N) * float64(r.K) / (r.MedianMs / 1e3) / 1e12

	// Error against the fp32 reference, computed on the device.
	out := f().AsType(mlx.DTypeFloat32)
	diff := out.Subtract(ref)
	num := mlx.Sum(mlx.Reshape(diff.Multiply(diff), -1), 0, false).Float()
	den := mlx.Sum(mlx.Reshape(ref.Multiply(ref), -1), 0, false).Float()
	r.RelRMS = math.Sqrt(float64(num) / math.Max(float64(den), 1e-30))
	r.MaxAbs = float64(mlx.Reshape(diff.Abs(), -1).MaxAxis(0, false).Float())
	mlx.Sweep()
}

// timeCalls makes warmup untimed calls, the first reported as FirstMs, then
// iters timed ones, and returns the timed durations. It stamps r with the
// wall-clock window of all of them, read from now.
func timeCalls(r *result, call func() float64, warmup, iters int, now func() time.Time) []float64 {
	r.StartUnixMs = now().UnixMilli()
	for i := range warmup {
		d := call()
		if i == 0 {
			r.FirstMs = d
		}
	}
	ts := make([]float64, iters)
	for i := range ts {
		ts[i] = call()
	}
	r.EndUnixMs = now().UnixMilli()
	return ts
}

// shapeHeader is the text log's line that opens a shape, stamped with the
// local time it started. The zone is printed too: in a container without TZ,
// local time is UTC.
func shapeHeader(sh shape, t time.Time) string {
	return fmt.Sprintf("\n## %s %s  K=%d N=%d  (w %d x %d, nvfp4 group %d)  at %s\n",
		sh.preset, sh.name, sh.k, sh.n, sh.n, sh.k, groupSize, t.Format("15:04:05 MST"))
}

func printRow(r result) {
	if r.Error != "" {
		fmt.Printf("   M=%-5d %-5s ERROR %s\n", r.M, r.Method, firstLine(r.Error))
		return
	}
	fmt.Printf("   M=%-5d %-5s median %8.3f ms  p10 %8.3f  p90 %8.3f  first %9.1f ms  %7.1f TFLOP/s  relRMS %.2e  maxAbs %.3g\n",
		r.M, r.Method, r.MedianMs, r.P10Ms, r.P90Ms, r.FirstMs, r.TFLOPS, r.RelRMS, r.MaxAbs)
}

// printTable renders one markdown table per shape: a row per M, a column per
// method with median ms and TFLOP/s, each other method's speed-up over qmm
// (qmm's median divided by the method's; above 1 is faster than today), and
// the relative RMS errors.
func printTable(results []result, methods []string) {
	byShape := map[string][]result{}
	var order []string
	for _, r := range results {
		key := r.Preset + " " + r.Proj + fmt.Sprintf(" (K=%d, N=%d)", r.K, r.N)
		if _, ok := byShape[key]; !ok {
			order = append(order, key)
		}
		byShape[key] = append(byShape[key], r)
	}
	fmt.Printf("\n# qqmm bench: median wall ms per call (TFLOP/s) after warm-up; relRMS vs fp32 · unquantized w\n")
	for _, key := range order {
		fmt.Printf("\n## %s\n\n| M |", key)
		for _, m := range methods {
			fmt.Printf(" %s ms (TFLOP/s) |", m)
		}
		var ratios []string
		for _, m := range methods {
			if m != "qmm" {
				ratios = append(ratios, m)
				fmt.Printf(" qmm / %s |", m)
			}
		}
		fmt.Printf(" relRMS %s |\n|---|", strings.Join(methods, " / "))
		for range methods {
			fmt.Printf("---|")
		}
		for range ratios {
			fmt.Printf("---|")
		}
		fmt.Printf("---|\n")
		rows := byShape[key]
		ms := map[int]map[string]result{}
		var mOrder []int
		for _, r := range rows {
			if _, ok := ms[r.M]; !ok {
				ms[r.M] = map[string]result{}
				mOrder = append(mOrder, r.M)
			}
			ms[r.M][r.Method] = r
		}
		for _, m := range mOrder {
			fmt.Printf("| %d |", m)
			var errs []string
			for _, method := range methods {
				r, ok := ms[m][method]
				switch {
				case !ok:
					fmt.Printf(" — |")
					errs = append(errs, "—")
				case r.Error != "":
					fmt.Printf(" error |")
					errs = append(errs, "error")
				default:
					fmt.Printf(" %.3f (%.0f) |", r.MedianMs, r.TFLOPS)
					errs = append(errs, fmt.Sprintf("%.1e", r.RelRMS))
				}
			}
			q, qok := ms[m]["qmm"]
			for _, name := range ratios {
				o, ok := ms[m][name]
				if qok && ok && q.Error == "" && o.Error == "" && o.MedianMs > 0 {
					fmt.Printf(" %.2fx |", q.MedianMs/o.MedianMs)
				} else {
					fmt.Printf(" — |")
				}
			}
			fmt.Printf(" %s |\n", strings.Join(errs, " / "))
		}
	}
}

func parseShapes(s string) ([]shape, error) {
	var shapes []shape
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if p, ok := presets[item]; ok {
			shapes = append(shapes, p...)
			continue
		}
		name, dims, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("unknown preset %q (known: %s); custom shapes are name=K:N", item, presetNames())
		}
		ks, ns, ok := strings.Cut(dims, ":")
		if !ok {
			return nil, fmt.Errorf("custom shape %q must be name=K:N", item)
		}
		k, err1 := strconv.Atoi(ks)
		n, err2 := strconv.Atoi(ns)
		if err1 != nil || err2 != nil || k <= 0 || n <= 0 {
			return nil, fmt.Errorf("custom shape %q must be name=K:N with positive integers", item)
		}
		shapes = append(shapes, shape{"custom", name, k, n})
	}
	if len(shapes) == 0 {
		return nil, fmt.Errorf("no shapes")
	}
	return shapes, nil
}

func presetNames() string {
	names := make([]string, 0, len(presets))
	for k := range presets {
		names = append(names, k)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, item := range strings.Split(s, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		v, err := strconv.Atoi(item)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("bad row count %q", item)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no row counts")
	}
	return out, nil
}

// randn fills n normal-distributed float32 values with the given standard
// deviation (Box–Muller; the host generates, MLX uploads).
func randn(rng *rand.Rand, n int, sd float32) []float32 {
	v := make([]float32, n)
	for i := 0; i < n; i += 2 {
		u1 := rng.Float64()
		if u1 < 1e-12 {
			u1 = 1e-12
		}
		u2 := rng.Float64()
		r := math.Sqrt(-2 * math.Log(u1))
		v[i] = float32(r*math.Cos(2*math.Pi*u2)) * sd
		if i+1 < n {
			v[i+1] = float32(r*math.Sin(2*math.Pi*u2)) * sd
		}
	}
	return v
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
