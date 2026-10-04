#!/usr/bin/env python3
"""Render a results.tsv from mmq-rocm-cases.sh or mmq-rocm-route.sh as Markdown: one table per mode, cases x forms of
903, each cell "pass" or "abort k/n" (runs that hit an illegal memory access), then the device's MMQ_DEBUG_PRINT line
per case and form.

    mmq-rocm-table.py <results.tsv> > matrix.md
"""
import collections, csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
okcol = 'passed' if 'passed' in rows[0] else 'ok'
variants = [v for v in ['none', 'main', 'widest', 'amended', 'head'] if any(r['variant'] == v for r in rows)]
label = {'none': 'b11081 as pinned (`ne11`)', 'main': "main's 903 (`ne12*n_used`)", 'widest': '903 widest tile',
         'amended': '903 amended (widest padded)', 'head': 'llama.cpp#29953 at 3070d927f'}
cases = list(dict.fromkeys(r['case'] for r in rows))
modes = list(dict.fromkeys(r['mode'] for r in rows))
cell = collections.defaultdict(lambda: [0, 0, 0])   # runs, aborts (illegal access), other failures
ids = {}
for r in rows:
    c = cell[(r['case'], r['variant'], r['mode'])]
    c[0] += 1
    if int(r['illegal_access'] or 0) > 0: c[1] += 1
    elif r[okcol] != '1': c[2] += 1
    if r.get('mmq_id'): ids.setdefault((r['case'], r['variant']), r['mmq_id'])
def fmt(c):
    n, a, f = c
    if n == 0: return '.'
    s = 'pass' if a == 0 and f == 0 else ('**abort %d/%d**' % (a, n) if a else '')
    if f: s += (' ' if s else '') + 'FAIL %d/%d' % (f, n)
    return s
for m in modes:
    print('\n#### %s\n' % m)
    print('| case | ' + ' | '.join(label[v] for v in variants) + ' |')
    print('|---|' + '---|' * len(variants))
    for c in cases:
        print('| `%s` | ' % c + ' | '.join(fmt(cell[(c, v, m)]) for v in variants) + ' |')
print('\n#### what the device reported (MMQ_DEBUG_PRINT, debug build)\n')
for c in cases:
    for v in variants:
        if (c, v) in ids: print('%-22s %-8s %s' % (c, v, ids[(c, v)]))
