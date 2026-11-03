#!/usr/bin/env python3
"""Run-local expectations for the v0.40.13 fold's canary: expectations.toml without the 0-40-2 CUDA profiles
(which claim the interim stamp 0.40.3-dynres-N), plus cuda-0-40-13 and mlx-cuda-0-40-13 cloned from 0-40-5 with
the b11963 / MLX 065ec468 pins and a version pattern admitting the interim stamp. The rows stay 0-40-5's until
measure_ladder.py says otherwise.  usage: exp-0413.py <expectations.toml> <out.toml> <stamp>"""
import re
import sys
import tomllib

src, dst, want = sys.argv[1:]
s = open(src).read()
a = s.index('[profiles.cuda-0-40-2]')
b = s.index('[profiles.cuda-0-40-5]')
s = s[:a] + s[b:]

# the 0-40-5 blocks run from [profiles.cuda-0-40-5] to the next profile that is not mlx-cuda-0-40-5, or the end
i = s.index('[profiles.cuda-0-40-5]')
j = len(s)
for m in re.finditer(r'^\[profiles\.([^\]]+)\]', s[i:], re.M):
    if m.group(1) not in ('cuda-0-40-5', 'mlx-cuda-0-40-5'):
        j = i + m.start()
        break
blk = s[i:j].replace('cuda-0-40-5', 'cuda-0-40-13')
pattern = r"^(0\.40\.13-dynres(\.\d+)?|0\.40\.3-dynres)-\d+-g[0-9a-f]{7,40}$"
blk = re.sub(r"^version_pattern = '.*'$", lambda _: f"version_pattern = '{pattern}'", blk, flags=re.M)
blk = blk.replace('llama_cpp_build = "620d2225c"', 'llama_cpp_build = "39febd2ad"')
blk = blk.replace('mlx_build = "bebd6422f8db88d56e6220520a35bcd146f823f8"',
                  'mlx_build = "065ec4683cce3c4d3d1d846b4759ad59c13a4ffd"')
s = s.rstrip('\n') + '\n\n' + blk
open(dst, 'w').write(s)

e = tomllib.loads(s)
for plat in ('cuda', 'mlx-cuda'):
    hits = [pid for pid, p in e['profiles'].items()
            if p['platform'] == plat and re.match(p['version_pattern'], want)]
    assert hits == [f'{plat}-0-40-13'], (plat, hits)
assert e['profiles']['cuda-0-40-13']['llama_cpp_build'] == '39febd2ad'
assert e['profiles']['mlx-cuda-0-40-13']['mlx_build'].startswith('065ec468')
assert e['profiles']['cuda-0-40-13']['mlx_build'].startswith('065ec468')
print('run-local expectations:', dst, sorted(k for k in e['expect'] if k.endswith('0-40-13')))
