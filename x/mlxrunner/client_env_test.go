package mlxrunner

import (
	"os"
	"os/exec"
	"strings"
	"testing"
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
			// t.Setenv cannot unset; restore whatever was there afterwards.
			if old, ok := os.LookupEnv(CacheThrashingCheckEnv); ok {
				t.Cleanup(func() { os.Setenv(CacheThrashingCheckEnv, old) })
			}
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
