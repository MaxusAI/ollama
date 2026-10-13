package mlxrunner

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func envEntries(cmd *exec.Cmd, key string) []string {
	var got []string
	for _, e := range cmd.Env {
		if strings.HasPrefix(e, key+"=") {
			got = append(got, strings.TrimPrefix(e, key+"="))
		}
	}
	return got
}

// MLX's graph-cache thrashing check is a performance advisory implemented as
// a throw out of graph commit, which the runner cannot survive
// (docs/maxusai/mlx-thrash-check-masks-as-cudagraph.md). The runner therefore
// starts with it off when nothing in the environment says otherwise.
func TestRunnerEnvDisablesThrashingCheckByDefault(t *testing.T) {
	for _, v := range []string{"", "unset"} {
		if v == "unset" {
			// t.Setenv registers the restore; the Unsetenv after it is the
			// state under test (usetesting forbids os.Setenv in tests).
			t.Setenv(CacheThrashingCheckEnv, "")
			os.Unsetenv(CacheThrashingCheckEnv)
		} else {
			t.Setenv(CacheThrashingCheckEnv, v)
		}
		cmd := &exec.Cmd{Env: os.Environ()}
		mlxRunnerEnvDefaults(cmd)
		if got := envEntries(cmd, CacheThrashingCheckEnv); len(got) != 1 || got[0] != "0" {
			t.Errorf("env %q: runner env should carry exactly one %s=0, got %v", v, CacheThrashingCheckEnv, got)
		}
	}
}

// An operator who wants MLX's advisory back exports the variable on the
// server; the default must not overwrite it.
func TestRunnerEnvKeepsOperatorThrashingCheck(t *testing.T) {
	t.Setenv(CacheThrashingCheckEnv, "1")
	cmd := &exec.Cmd{Env: os.Environ()}
	mlxRunnerEnvDefaults(cmd)
	if got := envEntries(cmd, CacheThrashingCheckEnv); len(got) != 1 || got[0] != "1" {
		t.Errorf("operator value must survive as the only entry, got %v", got)
	}
}

// admit prices gpus[0] only, so the runner must see only gpus[0]: with a
// second CUDA device visible, MLX's allocator fails host copies with
// "cudaMemcpy ... invalid argument" (see pinAdmittedDevice).
func TestRunnerEnvPinsTheAdmittedCUDADevice(t *testing.T) {
	cuda := func(id, filter string) ml.DeviceInfo {
		return ml.DeviceInfo{DeviceID: ml.DeviceID{ID: id, Library: "CUDA"}, FilterID: filter}
	}
	cases := []struct {
		name    string
		inherit string
		gpus    []ml.DeviceInfo
		want    string
	}{
		{"first of two", "", []ml.DeviceInfo{cuda("0", ""), cuda("1", "")}, "0"},
		{"replaces the server's own list", "0,1", []ml.DeviceInfo{cuda("0", ""), cuda("1", "")}, "0"},
		{"admitted device is not ordinal 0", "", []ml.DeviceInfo{cuda("1", ""), cuda("0", "")}, "1"},
		{"keeps the operator's numbering", "1,0", []ml.DeviceInfo{cuda("0", "1"), cuda("1", "0")}, "1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &exec.Cmd{}
			if tc.inherit != "" {
				cmd.Env = []string{"CUDA_VISIBLE_DEVICES=" + tc.inherit}
			}
			pinAdmittedDevice(cmd, tc.gpus)
			if got := envEntries(cmd, "CUDA_VISIBLE_DEVICES"); len(got) != 1 || got[0] != tc.want {
				t.Errorf("CUDA_VISIBLE_DEVICES = %v, want exactly [%s]", got, tc.want)
			}
		})
	}
}

// Metal has no visible-devices variable, and no device means nothing to pin.
func TestRunnerEnvLeavesNonCUDADevicesAlone(t *testing.T) {
	for _, gpus := range [][]ml.DeviceInfo{
		nil,
		{{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}},
	} {
		cmd := &exec.Cmd{Env: []string{"PATH=/usr/bin"}}
		pinAdmittedDevice(cmd, gpus)
		if len(cmd.Env) != 1 || cmd.Env[0] != "PATH=/usr/bin" {
			t.Errorf("gpus %v: env changed to %v", gpus, cmd.Env)
		}
	}
}
