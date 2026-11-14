#!/usr/bin/env python3
"""Run-local expectations for the v0.40.16 fold's canary: expectations.toml with cuda-0-40-13 and mlx-cuda-0-40-13
moved to cuda-0-40-16 and mlx-cuda-0-40-16. The fold's interim builds stamp 0.40.13-dynres-N (main's last tag is
v0.40.13-dynres), which the 0-40-13 pair claims, so the canary cannot run against the checked-in file. The clones
carry the b12177 / MLX 54a70511 pins, 908 out of the patchset (#505), and a version pattern admitting the interim
stamp. The rows stay 0-40-13's until measure_ladder.py says otherwise.
usage: exp-0416.py <expectations.toml> <out.toml> <stamp>"""
import re
import sys
import tomllib

src, dst, want = sys.argv[1:]
s = open(src).read()

# the 0-40-13 CUDA blocks run from [profiles.cuda-0-40-13] to the next profile that is not mlx-cuda-0-40-13
i = s.index('[profiles.cuda-0-40-13]')
j = len(s)
for m in re.finditer(r'^\[profiles\.([^\]]+)\]', s[i:], re.M):
    if m.group(1) not in ('cuda-0-40-13', 'mlx-cuda-0-40-13'):
        j = i + m.start()
        break
blk = s[i:j]
s = s[:i] + s[j:]
assert not re.search(r'^\[(profiles|expect)\.(mlx-)?cuda-0-40-13[\].]', s, re.M), 'a 0-40-13 CUDA table is left'

blk = blk.replace('cuda-0-40-13', 'cuda-0-40-16')
pattern = r"^(0\.40\.16-dynres(\.\d+)?|0\.40\.13-dynres)-\d+-g[0-9a-f]{7,40}$"
blk = re.sub(r"^version_pattern = '.*'$", lambda _: f"version_pattern = '{pattern}'", blk, flags=re.M)
blk = blk.replace('llama_cpp_build = "39febd2ad"', 'llama_cpp_build = "6c01da627"')
blk = blk.replace('mlx_build = "065ec4683cce3c4d3d1d846b4759ad59c13a4ffd"',
                  'mlx_build = "54a70511f4ea151378d03b657a4fdd823386ea38"')
blk = re.sub(r'^(patchset = \[.*), "908"\]$', r'\1]', blk, flags=re.M)
s = s.rstrip('\n') + '\n\n' + blk
open(dst, 'w').write(s)

e = tomllib.loads(s)
for plat in ('cuda', 'mlx-cuda'):
    hits = [pid for pid, p in e['profiles'].items()
            if p['platform'] == plat and re.match(p['version_pattern'], want)]
    assert hits == [f'{plat}-0-40-16'], (plat, hits)
for pid in ('cuda-0-40-16', 'mlx-cuda-0-40-16'):
    p = e['profiles'][pid]
    assert p.get('llama_cpp_build', '6c01da627') == '6c01da627', pid
    assert p['mlx_build'].startswith('54a70511'), pid
    assert '908' not in p.get('patchset', []), pid
assert e['profiles']['cuda-0-40-16']['llama_cpp_build'] == '6c01da627'
print('run-local expectations:', dst, sorted(k for k in e['expect'] if k.endswith('0-40-16')))
