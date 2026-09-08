package main

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
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
	for _, want := range []string{"## p q (K=64, N=128)", "| 8 | 1.000 (1) | 2.000 (0) | 1.000 (1) | 2.00x | 3.0e-03 / 1.0e-01 / 1.4e-01 |", "| 64 | 1.000 (1) | 2.000 (0) | error | — | 3.0e-03 / 1.0e-01 / error |"} {
		if !strings.Contains(out, want) {
			t.Errorf("table lacks %q:\n%s", want, out)
		}
	}
}
