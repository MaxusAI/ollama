### stock, memory pool, no sanitizer

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `ids16` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `p29847_b0` | 0/2 abort | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `p29847_b1` | 0/2 abort | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `j100_b0` | 2/2 | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `ids64` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `e120_b1` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `one113` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `fb65` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `orig2040` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `e120_b0` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `j100_b1` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `t100` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

### memory pool under memcheck

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `p29847_b0` | - | 1/1 E=0 | - | - | - | - | - |
| `p29847_b1` | - | 1/1 E=0 | - | - | - | - | - |
| `j100_b0` | - | 1/1 E=0 | - | - | - | - | - |

### exact-size cudaMalloc under memcheck

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `ids16` | 0/1 E=142 | 0/1 E=334 | 0/1 E=214 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `p29847_b0` | 0/1 E=838 | 0/1 E=776 | 0/1 E=772 | 0/1 E=820 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `p29847_b1` | 0/1 E=934 | 0/1 E=838 | 0/1 E=831 | 0/1 E=763 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `j100_b0` | 0/1 E=2489 | 0/1 E=2214 | 0/1 E=1021 | 0/1 E=1031 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `ids64` | 0/1 E=245 | 0/1 E=140 | 0/1 E=166 | 0/1 E=170 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `e120_b1` | 0/1 E=1332 | 0/1 E=1428 | 0/1 E=1470 | 0/1 E=1537 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `one113` | 0/1 E=2594 | 0/1 E=2594 | 0/1 E=11939 | 0/1 E=2639 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `fb65` | 0/1 E=523 | 0/1 E=324 | 0/1 E=280 | 0/1 E=430 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `orig2040` | 0/1 E=46 | 0/1 E=57 | 0/1 E=43 | 0/1 E=58 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `e120_b0` | 0/1 E=1592 | 0/1 E=1693 | 0/1 E=1434 | 0/1 E=1367 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `j100_b1` | 0/1 E=3120 | 0/1 E=3058 | 0/1 E=2594 | 0/1 E=2554 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |
| `t100` | 0/1 E=2150 | 0/1 E=1630 | 0/1 E=1498 | 0/1 E=1825 | 1/1 E=0 | 1/1 E=0 | 1/1 E=0 |

### guard page on src1 and ids_dst, no sanitizer

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `ids16` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `p29847_b0` | - | 0/3 abort | - | - | - | - | - |
| `p29847_b1` | - | 0/3 abort | - | - | - | - | - |
| `j100_b0` | 0/2 abort | 0/3 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `ids64` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `e120_b1` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `one113` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |

### guard page on ids_dst only, no sanitizer

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `ids16` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `p29847_b0` | 0/2 abort | 0/3 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `p29847_b1` | 0/2 abort | 0/3 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `j100_b0` | 0/2 abort | 0/3 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `ids64` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `e120_b1` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `one113` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `fb65` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `orig2040` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `e120_b0` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `j100_b1` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `t100` | 0/2 abort | 0/2 abort | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 |

### guard page on src1 only, no sanitizer

| case | ne11 (#24127) | ne12 (#29941, master) | #29953 | ne12*n_expert_used (#27044) | widest tile (#448) | widest padded tile (#448 amended) | #29953 + amendment |
|---|---|---|---|---|---|---|---|
| `ids16` | 0/2 abort | 0/10 abort | 0/10 abort | 2/2 | 2/2 | 2/2 | 10/10 |
| `p29847_b0` | 0/2 abort | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `p29847_b1` | 0/2 abort | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `j100_b0` | 0/2 abort | 0/10 abort | 9/10 abort | 2/2 | 2/2 | 2/2 | 10/10 |
| `ids64` | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `e120_b1` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `one113` | 0/2 abort | 0/2 abort | 2/2 | 0/2 abort | 2/2 | 2/2 | 2/2 |
| `fb65` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `orig2040` | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `e120_b0` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `j100_b1` | 0/2 abort | 0/10 abort | 6/10 abort | 2/2 | 2/2 | 2/2 | 10/10 |
| `t100` | 0/2 abort | 0/2 abort | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

