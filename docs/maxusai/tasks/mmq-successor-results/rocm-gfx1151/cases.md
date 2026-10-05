
#### stock

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `p29847_b0` | pass | pass | pass | pass |
| `p29847_b1` | pass | pass | pass | pass |
| `j100_b0` | pass | pass | pass | pass |
| `j100_b1` | pass | pass | pass | pass |
| `e120_b0` | pass | pass | pass | pass |
| `e120_b1` | pass | pass | pass | pass |
| `one113` | pass | pass | pass | pass |
| `orig2040` | pass | pass | pass | pass |
| `fb65` | pass | pass | pass | pass |
| `t100` | pass | pass | pass | pass |
| `ids16` | pass | pass | pass | pass |
| `ids64` | pass | pass | pass | pass |

#### guard_ids_dst

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `p29847_b0` | **abort 3/3** | **abort 3/3** | pass | pass |
| `p29847_b1` | **abort 3/3** | **abort 3/3** | pass | pass |
| `j100_b0` | **abort 3/3** | **abort 3/3** | pass | pass |
| `j100_b1` | **abort 3/3** | **abort 3/3** | pass | pass |
| `e120_b0` | **abort 3/3** | **abort 3/3** | pass | pass |
| `e120_b1` | **abort 3/3** | **abort 3/3** | pass | pass |
| `one113` | **abort 3/3** | **abort 3/3** | pass | pass |
| `orig2040` | **abort 3/3** | **abort 3/3** | pass | pass |
| `fb65` | **abort 3/3** | **abort 3/3** | pass | pass |
| `t100` | **abort 3/3** | **abort 3/3** | pass | pass |
| `ids16` | **abort 3/3** | **abort 3/3** | pass | pass |
| `ids64` | **abort 3/3** | **abort 3/3** | pass | pass |

#### guard_src1

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `p29847_b0` | **abort 3/3** | pass | pass | pass |
| `p29847_b1` | **abort 3/3** | pass | pass | pass |
| `j100_b0` | **abort 3/3** | pass | pass | pass |
| `j100_b1` | **abort 3/3** | pass | pass | pass |
| `e120_b0` | **abort 3/3** | pass | pass | pass |
| `e120_b1` | **abort 3/3** | pass | pass | pass |
| `one113` | **abort 3/3** | pass | pass | pass |
| `orig2040` | **abort 2/3** | pass | pass | pass |
| `fb65` | **abort 3/3** | pass | pass | pass |
| `t100` | **abort 3/3** | pass | pass | pass |
| `ids16` | **abort 1/3** | pass | pass | pass |
| `ids64` | pass | pass | pass | pass |

#### what the device reported (MMQ_DEBUG_PRINT, debug build)

p29847_b0              none     MMQ_ID q4_0 fb=0 ne11= 10 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 5080 need= 15 pad= 0 SHORT 
p29847_b0              main     MMQ_ID q4_0 fb=0 ne11= 10 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad= 0 SHORT 
p29847_b0              widest   MMQ_ID q4_0 fb=0 ne11= 10 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad=128 covered
p29847_b0              amended  MMQ_ID q4_0 fb=0 ne11= 10 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad=128 covered
p29847_b1              none     MMQ_ID q4_0 fb=0 ne11= 1 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 5080 need= 15 pad= 0 SHORT 
p29847_b1              main     MMQ_ID q4_0 fb=0 ne11= 1 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad= 0 SHORT 
p29847_b1              widest   MMQ_ID q4_0 fb=0 ne11= 1 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad=128 covered
p29847_b1              amended  MMQ_ID q4_0 fb=0 ne11= 1 ne12= 508 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 5080 need= 15 pad=128 covered
j100_b0                none     MMQ_ID q4_0 fb=0 ne11= 10 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 1000 need= 15 pad= 0 SHORT 
j100_b0                main     MMQ_ID q4_0 fb=0 ne11= 10 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad= 0 SHORT 
j100_b0                widest   MMQ_ID q4_0 fb=0 ne11= 10 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad=128 covered
j100_b0                amended  MMQ_ID q4_0 fb=0 ne11= 10 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad=128 covered
j100_b1                none     MMQ_ID q4_0 fb=0 ne11= 1 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 1000 need= 15 pad= 0 SHORT 
j100_b1                main     MMQ_ID q4_0 fb=0 ne11= 1 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad= 0 SHORT 
j100_b1                widest   MMQ_ID q4_0 fb=0 ne11= 1 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad=128 covered
j100_b1                amended  MMQ_ID q4_0 fb=0 ne11= 1 ne12= 100 n_used= 10 ne02= 512 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1000 need= 15 pad=128 covered
e120_b0                none     MMQ_ID q4_K fb=1 ne11= 10 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 1200 need= 15 pad= 0 SHORT 
e120_b0                main     MMQ_ID q4_K fb=1 ne11= 10 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad= 0 SHORT 
e120_b0                widest   MMQ_ID q4_K fb=1 ne11= 10 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad=128 covered
e120_b0                amended  MMQ_ID q4_K fb=1 ne11= 10 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad=128 covered
e120_b1                none     MMQ_ID q4_K fb=1 ne11= 1 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 1200 need= 15 pad= 0 SHORT 
e120_b1                main     MMQ_ID q4_K fb=1 ne11= 1 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad= 0 SHORT 
e120_b1                widest   MMQ_ID q4_K fb=1 ne11= 1 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad=128 covered
e120_b1                amended  MMQ_ID q4_K fb=1 ne11= 1 ne12= 120 n_used= 10 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 1200 need= 15 pad=128 covered
one113                 none     MMQ_ID q4_K fb=1 ne11= 1 ne12= 113 n_used= 1 ne02= 64 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 113 need= 15 pad= 0 SHORT 
one113                 main     MMQ_ID q4_K fb=1 ne11= 1 ne12= 113 n_used= 1 ne02= 64 | J= 16 nthreads=128 | src1 need= 17 pad= 64 covered | ids_dst n= 113 need= 15 pad= 0 SHORT 
one113                 widest   MMQ_ID q4_K fb=1 ne11= 1 ne12= 113 n_used= 1 ne02= 64 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 113 need= 15 pad=128 covered
one113                 amended  MMQ_ID q4_K fb=1 ne11= 1 ne12= 113 n_used= 1 ne02= 64 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 113 need= 15 pad=128 covered
orig2040               none     MMQ_ID q4_K fb=0 ne11= 1 ne12= 2040 n_used= 8 ne02= 256 | J= 64 nthreads=256 | src1 need= 63 pad= 0 SHORT | ids_dst n= 16320 need= 63 pad= 0 SHORT 
orig2040               main     MMQ_ID q4_K fb=0 ne11= 1 ne12= 2040 n_used= 8 ne02= 256 | J= 64 nthreads=256 | src1 need= 63 pad=128 covered | ids_dst n= 16320 need= 63 pad= 0 SHORT 
orig2040               widest   MMQ_ID q4_K fb=0 ne11= 1 ne12= 2040 n_used= 8 ne02= 256 | J= 64 nthreads=256 | src1 need= 63 pad=128 covered | ids_dst n= 16320 need= 63 pad=128 covered
orig2040               amended  MMQ_ID q4_K fb=0 ne11= 1 ne12= 2040 n_used= 8 ne02= 256 | J= 64 nthreads=256 | src1 need= 63 pad=128 covered | ids_dst n= 16320 need= 63 pad=128 covered
fb65                   none     MMQ_ID q4_K fb=1 ne11= 1 ne12= 65 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 520 need= 15 pad= 0 SHORT 
fb65                   main     MMQ_ID q4_K fb=1 ne11= 1 ne12= 65 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 520 need= 15 pad= 0 SHORT 
fb65                   widest   MMQ_ID q4_K fb=1 ne11= 1 ne12= 65 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 520 need= 15 pad=128 covered
fb65                   amended  MMQ_ID q4_K fb=1 ne11= 1 ne12= 65 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 520 need= 15 pad=128 covered
t100                   none     MMQ_ID q4_K fb=0 ne11= 1 ne12= 100 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 800 need= 15 pad= 0 SHORT 
t100                   main     MMQ_ID q4_K fb=0 ne11= 1 ne12= 100 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 800 need= 15 pad= 0 SHORT 
t100                   widest   MMQ_ID q4_K fb=0 ne11= 1 ne12= 100 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 800 need= 15 pad=128 covered
t100                   amended  MMQ_ID q4_K fb=0 ne11= 1 ne12= 100 n_used= 8 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 800 need= 15 pad=128 covered
ids16                  none     MMQ_ID q4_0 fb=0 ne11= 16 ne12= 16 n_used= 16 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad= 16 SHORT | ids_dst n= 256 need= 15 pad= 0 SHORT 
ids16                  main     MMQ_ID q4_0 fb=0 ne11= 16 ne12= 16 n_used= 16 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 256 need= 15 pad= 0 SHORT 
ids16                  widest   MMQ_ID q4_0 fb=0 ne11= 16 ne12= 16 n_used= 16 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 256 need= 15 pad=128 covered
ids16                  amended  MMQ_ID q4_0 fb=0 ne11= 16 ne12= 16 n_used= 16 ne02= 256 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 256 need= 15 pad=128 covered
ids64                  none     MMQ_ID q4_0 fb=0 ne11= 64 ne12= 100 n_used= 64 ne02= 256 | J= 32 nthreads=128 | src1 need= 31 pad= 64 covered | ids_dst n= 6400 need= 31 pad= 0 SHORT 
ids64                  main     MMQ_ID q4_0 fb=0 ne11= 64 ne12= 100 n_used= 64 ne02= 256 | J= 32 nthreads=128 | src1 need= 31 pad=128 covered | ids_dst n= 6400 need= 31 pad= 0 SHORT 
ids64                  widest   MMQ_ID q4_0 fb=0 ne11= 64 ne12= 100 n_used= 64 ne02= 256 | J= 32 nthreads=128 | src1 need= 31 pad=128 covered | ids_dst n= 6400 need= 31 pad=128 covered
ids64                  amended  MMQ_ID q4_0 fb=0 ne11= 64 ne12= 100 n_used= 64 ne02= 256 | J= 32 nthreads=128 | src1 need= 31 pad=128 covered | ids_dst n= 6400 need= 31 pad=128 covered
