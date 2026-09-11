package main

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestParseShapesPresetsAndCustom(t *testing.T) {
	shapes, err := parseShapes("gemma4-31b, custom=1024:2048")
	if err != nil {
		t.Fatal(err)
	}
	if len(shapes) != len(presets["gemma4-31b"])+1 {
		t.Fatalf("got %d shapes", len(shapes))
	}
	last := shapes[len(shapes)-1]
	if last.preset != "custom" || last.name != "custom" || last.k != 1024 || last.n != 2048 {
		t.Fatalf("custom shape parsed as %+v", last)
	}
	for _, sh := range presets["gemma4-31b"] {
		// The prefill kernel needs n % 128 == 0 and k % 64 == 0; the presets must qualify.
		if sh.n%128 != 0 || sh.k%64 != 0 {
			t.Errorf("%s %s: n=%d k=%d would be refused by qmm_sm80", sh.preset, sh.name, sh.n, sh.k)
		}
	}
}

func TestParseShapesRejectsBadInput(t *testing.T) {
	for _, in := range []string{"", "nope", "x=1024", "x=a:b", "x=0:5"} {
		if _, err := parseShapes(in); err == nil {
			t.Errorf("%q: expected an error", in)
		}
	}
}

func TestParseInts(t *testing.T) {
	got, err := parseInts("1, 8,64")
	if err != nil || len(got) != 3 || got[0] != 1 || got[1] != 8 || got[2] != 64 {
		t.Fatalf("got %v, %v", got, err)
	}
	for _, in := range []string{"", "0", "a"} {
		if _, err := parseInts(in); err == nil {
			t.Errorf("%q: expected an error", in)
		}
	}
}

func TestRandnMoments(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	v := randn(rng, 200001, 0.5) // odd length exercises the tail branch
	var sum, sq float64
	for _, x := range v {
		sum += float64(x)
		sq += float64(x) * float64(x)
	}
	mean := sum / float64(len(v))
	sd := math.Sqrt(sq/float64(len(v)) - mean*mean)
	if math.Abs(mean) > 0.01 || math.Abs(sd-0.5) > 0.01 {
		t.Fatalf("mean %.4f sd %.4f", mean, sd)
	}
}

func TestPrintTableRendersSpeedupAndErrors(t *testing.T) {
	results := []result{
		{Preset: "p", Proj: "q", M: 8, K: 64, N: 128, Method: "bf16", MedianMs: 1, TFLOPS: 1, RelRMS: 0.003},
		{Preset: "p", Proj: "q", M: 8, K: 64, N: 128, Method: "qmm", MedianMs: 2, TFLOPS: 0.5, RelRMS: 0.1},
		{Preset: "p", Proj: "q", M: 8, K: 64, N: 128, Method: "qqmm", MedianMs: 1, TFLOPS: 1, RelRMS: 0.14},
		{Preset: "p", Proj: "q", M: 64, K: 64, N: 128, Method: "bf16", MedianMs: 1, TFLOPS: 1, RelRMS: 0.003},
		{Preset: "p", Proj: "q", M: 64, K: 64, N: 128, Method: "qmm", MedianMs: 2, TFLOPS: 0.5, RelRMS: 0.1},
		{Preset: "p", Proj: "q", M: 64, K: 64, N: 128, Method: "qqmm", Error: "unsupported"},
	}
	out := captureStdout(t, func() { printTable(results, []string{"bf16", "qmm", "qqmm"}) })
	for _, want := range []string{"## p q (K=64, N=128)", "| qmm / bf16 | qmm / qqmm |", "| 8 | 1.000 (1) | 2.000 (0) | 1.000 (1) | 2.00x | 2.00x | 3.0e-03 / 1.0e-01 / 1.4e-01 |", "| 64 | 1.000 (1) | 2.000 (0) | error | 2.00x | — | 3.0e-03 / 1.0e-01 / error |"} {
		if !strings.Contains(out, want) {
			t.Errorf("table lacks %q:\n%s", want, out)
		}
	}
}

// jsonKeys returns the top-level keys of v's JSON encoding in the order written.
func jsonKeys(t *testing.T, v any) []string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	if _, err := dec.Token(); err != nil {
		t.Fatal(err)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			t.Fatal(err)
		}
		keys = append(keys, tok.(string))
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			t.Fatal(err)
		}
	}
	return keys
}

func TestResultJSONAppendsTheWindowAfterTheExistingFields(t *testing.T) {
	existing := []string{"preset", "proj", "m", "k", "n", "method", "first_ms", "median_ms", "p10_ms", "p90_ms", "tflops", "rel_rms_err", "max_abs_err", "peak_gib"}
	r := result{Preset: "p", Proj: "q", M: 8, K: 64, N: 128, Method: "qmm", StartUnixMs: 1789115000123, EndUnixMs: 1789115000456}
	if got, want := jsonKeys(t, r), append(slices.Clone(existing), "start_unix_ms", "end_unix_ms"); !slices.Equal(got, want) {
		t.Fatalf("keys %v, want %v", got, want)
	}
	r.Error = "unsupported"
	if got, want := jsonKeys(t, r), append(slices.Clone(existing), "error", "start_unix_ms", "end_unix_ms"); !slices.Equal(got, want) {
		t.Fatalf("keys of an error row %v, want %v", got, want)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"start_unix_ms":1789115000123,"end_unix_ms":1789115000456`) {
		t.Fatalf("window not written as integer milliseconds: %s", b)
	}
}

func TestTimeCallsStampsTheWindowAroundTheMeasuredCalls(t *testing.T) {
	base := time.UnixMilli(1789115000000)
	ticks := 0
	now := func() time.Time { ticks++; return base.Add(time.Duration(ticks) * time.Millisecond) }
	var callAt []int64
	n := 0
	call := func() float64 {
		callAt = append(callAt, now().UnixMilli())
		n++
		return float64(n) // the n-th call takes n ms
	}

	var r result
	ts := timeCalls(&r, call, 3, 5, now)
	if len(callAt) != 8 {
		t.Fatalf("%d calls, want 3 warm-up + 5 timed", len(callAt))
	}
	if r.FirstMs != 1 {
		t.Errorf("first_ms %v, want the first warm-up call's 1", r.FirstMs)
	}
	if want := []float64{4, 5, 6, 7, 8}; !slices.Equal(ts, want) {
		t.Errorf("timed %v, want %v", ts, want)
	}
	if r.StartUnixMs >= callAt[0] || r.EndUnixMs <= callAt[len(callAt)-1] {
		t.Errorf("window %d..%d does not enclose the calls %d..%d", r.StartUnixMs, r.EndUnixMs, callAt[0], callAt[len(callAt)-1])
	}

	var r0 result
	callAt = nil
	ts = timeCalls(&r0, call, 0, 2, now)
	if len(ts) != 2 || r0.FirstMs != 0 || r0.StartUnixMs >= callAt[0] || r0.EndUnixMs <= callAt[1] {
		t.Errorf("without warm-up: timed %v, first_ms %v, window %d..%d around calls %v", ts, r0.FirstMs, r0.StartUnixMs, r0.EndUnixMs, callAt)
	}
}

func TestShapeHeaderStampsTheStartTime(t *testing.T) {
	sh := shape{"gemma4-31b", "q_proj", 5376, 8192}
	at := time.Date(2026, 9, 11, 16, 47, 25, 0, time.FixedZone("AEST", 10*3600))
	if got, want := shapeHeader(sh, at), "\n## gemma4-31b q_proj  K=5376 N=8192  (w 8192 x 5376, nvfp4 group 16)  at 16:47:25 AEST\n"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got := shapeHeader(sh, at.UTC()); !strings.HasSuffix(got, "  at 06:47:25 UTC\n") {
		t.Fatalf("a UTC clock must say so: %q", got)
	}
}
