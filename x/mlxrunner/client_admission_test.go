package mlxrunner

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/mlxrunner/kvsize"
)

// The rungs the vision suite's context ladder climbs. Before admission priced
// the rung these four admitted identically on MLX: the KV cache was never in
// the comparison, so a rung that could not fit was accepted and then killed
// mid-prefill (docs/maxusai/mlx-admission-prices-weights-only.md).
var admissionLadder = []int{8192, 16384, 32768, 65536}

// fakeEstimate stands in for kvsize on a synthetic model: bytesPerToken of
// full-attention KV per token, so a rung's KV is exactly numCtx*bytesPerToken
// and the expected numbers stay arithmetic. Injecting it keeps these tests off
// the manifest and off the disk.
func fakeEstimate(bytesPerToken uint64) func(int) kvsize.Estimate {
	return func(numCtx int) kvsize.Estimate {
		return kvsize.Estimate{
			Known:     true,
			Arch:      "FakeForCausalLM",
			NumCtx:    numCtx,
			ElemBytes: 2,
			Attention: uint64(numCtx) * bytesPerToken,
			Layers:    kvsize.LayerCounts{Attention: 1},
		}
	}
}

func admissionClient(weights uint64, estimate func(int) kvsize.Estimate, numCtx int, auto bool) *Client {
	c := &Client{modelName: "fake:test", numCtxAuto: auto, kvEstimate: estimate}
	c.memory.Store(weights)
	c.softContextLength.Store(int64(numCtx))
	return c
}

// gpuWith returns a device whose usable memory (after the overhead admission
// subtracts) is exactly want.
func gpuWith(want uint64) []ml.DeviceInfo {
	var d ml.DeviceInfo
	return []ml.DeviceInfo{{FreeMemory: want + d.MinimumMemory() + envconfig.GpuOverhead()}}
}

// (a) The rung has to be visible to admission. Each rung must price higher than
// the one below it, and a card sized between two of them must admit the lower
// pair and refuse the upper pair.
//
// This test fails on the pre-change code: pricing weights alone, all four rungs
// produce the same number and all four admit.
func TestLadderRungsAdmitDifferently(t *testing.T) {
	const (
		weights       = 40 << 30
		bytesPerToken = 64 << 10 // 512 MiB of KV at 8192, 4 GiB at 65536
	)
	c := admissionClient(weights, fakeEstimate(bytesPerToken), admissionLadder[0], false)

	needs := make([]uint64, len(admissionLadder))
	for i, rung := range admissionLadder {
		needs[i], _, _, _ = c.needFor(weights, rung)
		if i > 0 && needs[i] <= needs[i-1] {
			t.Fatalf("num_ctx %d needs %d, not more than %d at %d: the rung is invisible to admission",
				rung, needs[i], needs[i-1], admissionLadder[i-1])
		}
	}

	// A card that sits between the second and third rung: the first two fit,
	// the last two do not.
	gpus := gpuWith((needs[1] + needs[2]) / 2)
	for i, rung := range admissionLadder {
		c.softContextLength.Store(int64(rung))
		_, err := c.admit(gpus, false)
		if i < 2 {
			if err != nil {
				t.Errorf("num_ctx %d needs %s and must still admit: %v", rung, format.HumanBytes2(needs[i]), err)
			}
			continue
		}
		if err == nil {
			t.Errorf("num_ctx %d needs %s, more than the card has, and must be refused", rung, format.HumanBytes2(needs[i]))
			continue
		}
		if !strings.Contains(err.Error(), strconv.Itoa(rung)) {
			t.Errorf("num_ctx %d: refusal must name the rung, got: %v", rung, err)
		}
	}
}

// (b) An explicit rung that does not fit is a refusal, not a clamp: the caller
// asked for that window and silently serving a smaller one would be a lie.
// Under requireFull it stays evictable, because freeing another runner raises
// FreeMemory and a retry can then succeed.
func TestExplicitRungThatDoesNotFitIsRefused(t *testing.T) {
	const weights = 40 << 30
	c := admissionClient(weights, fakeEstimate(64<<10), 65536, false)
	need, kv, headroom, _ := c.needFor(weights, 65536)
	gpus := gpuWith(need - 1)

	if _, err := c.admit(gpus, true); !errors.Is(err, llm.ErrLoadRequiredFull) {
		t.Errorf("requireFull: got %v, want ErrLoadRequiredFull so the scheduler can evict and retry", err)
	}

	_, err := c.admit(gpus, false)
	if err == nil {
		t.Fatal("expected a refusal")
	}
	for _, want := range []string{
		"65536",                                    // the rung
		format.HumanBytes2(weights),                // the weights
		format.HumanBytes2(kv),                     // the KV cache
		format.HumanBytes2(headroom),               // the headroom
		"lower num_ctx or free VRAM on the device", // what to do about it
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal must contain %q, got: %v", want, err)
		}
	}

	// The rung must not have been clamped behind the caller's back.
	if got := c.reportedContextLength(0); got != 65536 {
		t.Errorf("served context = %d, want the requested 65536: an explicit rung is refused, never clamped", got)
	}
}

// (c) An automatic rung was never asked for -- it is Ollama's VRAM-tier default
// (262144 on the CUDA host) -- so pricing it and refusing would refuse
// everything. It steps down until it fits and serves the smaller window.
func TestAutomaticRungIsClampedNotRefused(t *testing.T) {
	const weights = 40 << 30
	c := admissionClient(weights, fakeEstimate(64<<10), 262144, true)

	// Size the card so 32768 fits and 65536 does not.
	need32k, _, _, _ := c.needFor(weights, 32768)
	need64k, _, _, _ := c.needFor(weights, 65536)
	gpus := gpuWith((need32k + need64k) / 2)

	if _, err := c.admit(gpus, false); err != nil {
		t.Fatalf("an automatic rung must clamp, not refuse: %v", err)
	}
	if got := c.reportedContextLength(0); got != 32768 {
		t.Errorf("served context = %d, want it clamped to the 32768 that fits", got)
	}
}

// ...and when even the floor does not fit, an automatic rung still admits on
// the weights alone. The user cannot act on a refusal for a window they never
// chose, and the operator cap plus the runner's own ceiling still bound it.
func TestAutomaticRungAdmitsOnWeightsWhenNothingFits(t *testing.T) {
	const weights = 40 << 30
	c := admissionClient(weights, fakeEstimate(64<<10), 262144, true)

	// Room for the weights and nothing more.
	if _, err := c.admit(gpuWith(weights), false); err != nil {
		t.Fatalf("weights fit, so an automatic rung must admit: %v", err)
	}
	if got := c.reportedContextLength(0); got != autoContextFloor {
		t.Errorf("served context = %d, want the floor %d", got, autoContextFloor)
	}

	// Weights that genuinely do not fit are still a physical shortfall.
	if _, err := c.admit(gpuWith(weights-1), true); !errors.Is(err, llm.ErrLoadRequiredFull) {
		t.Errorf("weights alone over the card: got %v, want ErrLoadRequiredFull", err)
	}
}

// (d) The operator's ceiling still wins, and the KV is inside what it bounds.
// It must NOT be ErrLoadRequiredFull: eviction raises FreeMemory, and the cap
// is a constant free memory cannot move.
func TestOperatorCapCoversTheContextRung(t *testing.T) {
	const weights = 40 << 30
	c := admissionClient(weights, fakeEstimate(64<<10), 65536, false)
	need, _, _, _ := c.needFor(weights, 65536)

	// The card has room; the operator's cap does not. The cap is above the
	// weights, so before the rung was priced this admitted.
	t.Setenv(MemoryLimitEnv, strconv.FormatUint(need-1, 10))
	gpus := gpuWith(need + (10 << 30))

	_, err := c.admit(gpus, true)
	if err == nil {
		t.Fatal("expected a refusal from the operator cap")
	}
	if errors.Is(err, llm.ErrLoadRequiredFull) {
		t.Fatal("an operator cap must not request eviction; eviction cannot lower a constant")
	}
	if !strings.Contains(err.Error(), MemoryLimitEnv) {
		t.Errorf("error must name the variable the operator has to change, got: %v", err)
	}
}

// (e) An architecture the estimator has no rule for keeps today's behaviour
// exactly: weights against available, no headroom, no refusal it did not make
// before. A wrong estimate that refuses a servable model is worse than the
// over-admission we are fixing.
func TestUnknownArchitectureKeepsWeightsOnlyAdmission(t *testing.T) {
	const weights = 40 << 30
	unknown := func(numCtx int) kvsize.Estimate {
		return kvsize.Estimate{Arch: "SomethingNewForCausalLM", NumCtx: numCtx}
	}

	for _, rung := range admissionLadder {
		c := admissionClient(weights, unknown, rung, false)
		if _, err := c.admit(gpuWith(weights), false); err != nil {
			t.Errorf("num_ctx %d: weights fit exactly, so an unpriced arch must admit: %v", rung, err)
		}
		c = admissionClient(weights, unknown, rung, false)
		_, err := c.admit(gpuWith(weights-1), false)
		if err == nil {
			t.Errorf("num_ctx %d: weights over the card must still refuse", rung)
			continue
		}
		if strings.Contains(err.Error(), "num_ctx") {
			t.Errorf("num_ctx %d: an unpriced arch must not blame the rung, got: %v", rung, err)
		}
	}

	// A client built with no readable config behaves the same way.
	c := admissionClient(weights, nil, 65536, false)
	if _, err := c.admit(gpuWith(weights), false); err != nil {
		t.Errorf("no estimator at all must admit exactly as before: %v", err)
	}
}

// Load has to call admission, not merely contain it: the checks above pass even
// if the call site is deleted.
func TestLoadRefusesARungThatDoesNotFit(t *testing.T) {
	const weights = 40 << 30
	c := admissionClient(weights, fakeEstimate(64<<10), 65536, false)
	need, _, _, _ := c.needFor(weights, 65536)

	_, err := c.Load(t.Context(), ml.SystemInfo{}, gpuWith(need-1), false)
	if err == nil {
		t.Fatal("expected Load to refuse a rung that does not fit")
	}
	if !strings.Contains(err.Error(), "num_ctx 65536") {
		t.Errorf("Load must surface the priced refusal, got: %v", err)
	}
}

// The headroom is a placeholder to be calibrated on GPU; what must not drift is
// its shape -- a floor for small models, a fraction for large ones.
func TestHeadroomIsAFloorThenAFraction(t *testing.T) {
	const floor = 512 << 20
	if got := admissionHeadroom(1 << 30); got != floor {
		t.Errorf("1 GiB model: headroom %d, want the %d floor", got, floor)
	}
	if got := admissionHeadroom(40 << 30); got != (40<<30)/20 {
		t.Errorf("40 GiB model: headroom %d, want 5%%", got)
	}
}
