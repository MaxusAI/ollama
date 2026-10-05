
#### stock

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `p29847_b0` | pass |
| `p29847_b1` | pass |
| `j100_b0` | pass |
| `j100_b1` | pass |
| `e120_b0` | pass |
| `e120_b1` | pass |
| `one113` | pass |
| `orig2040` | pass |
| `fb65` | pass |
| `t100` | pass |
| `ids16` | pass |
| `ids64` | pass |
| `dense321` | pass |

#### guard_ids_dst

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `p29847_b0` | pass |
| `p29847_b1` | pass |
| `j100_b0` | pass |
| `j100_b1` | pass |
| `e120_b0` | pass |
| `e120_b1` | pass |
| `one113` | pass |
| `orig2040` | pass |
| `fb65` | pass |
| `t100` | pass |
| `ids16` | pass |
| `ids64` | pass |
| `dense321` | pass |

#### guard_src1

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `p29847_b0` | pass |
| `p29847_b1` | pass |
| `j100_b0` | pass |
| `j100_b1` | pass |
| `e120_b0` | pass |
| `e120_b1` | pass |
| `one113` | pass |
| `orig2040` | pass |
| `fb65` | pass |
| `t100` | pass |
| `ids16` | pass |
| `ids64` | pass |
| `dense321` | pass |

#### what the device reported (MMQ_DEBUG_PRINT, debug build)

p29847_b0              head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 10 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 5080 need= 15 pad= 15 covered
p29847_b1              head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 1 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 5080 need= 15 pad= 15 covered
j100_b0                head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 10 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 1000 need= 15 pad= 15 covered
j100_b1                head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 1 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 1000 need= 15 pad= 15 covered
e120_b0                head     MMQ_IDS q4_K fb=1 prec=q8 ne11= 10 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 1200 need= 15 pad= 15 covered
e120_b1                head     MMQ_IDS q4_K fb=1 prec=q8 ne11= 1 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 1200 need= 15 pad= 15 covered
one113                 head     MMQ_IDS q4_K fb=1 prec=q8 ne11= 1 ne12= 113 n_used= 1 ne02= 64 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 113 need= 15 pad= 15 covered
orig2040               head     MMQ_IDS q4_K fb=0 prec=q8 ne11= 1 ne12= 2040 n_used= 8 ne02= 256 | J= 64 nthreads=256 | src1 need= 63 pad= 64 covered | ids_dst n= 16320 need= 63 pad= 63 covered
fb65                   head     MMQ_IDS q4_K fb=1 prec=q8 ne11= 1 ne12= 65 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 520 need= 15 pad= 15 covered
t100                   head     MMQ_IDS q4_K fb=0 prec=q8 ne11= 1 ne12= 100 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 800 need= 15 pad= 15 covered
ids16                  head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 16 ne12= 16 n_used= 16 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 256 need= 15 pad= 15 covered
ids64                  head     MMQ_IDS q4_0 fb=0 prec=q8 ne11= 64 ne12= 100 n_used= 64 ne02= 256 | J= 32 nthreads=128 | src1 need= 31 pad= 32 covered | ids_dst n= 6400 need= 31 pad= 31 covered
