
#### print

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `q2k_e8_last1_b0` | pass | pass | pass | pass |
| `q2k_e8_last1_b1` | pass | pass | pass | pass |
| `q2k_e8_unif_b0` | pass | pass | pass | pass |
| `q2k_e128_last1_b1` | pass | pass | pass | pass |
| `q2k_e128_unif_b1` | pass | pass | pass | pass |
| `q4k_e128_u6_n5_b1` | pass | pass | pass | pass |
| `chk_65_8_8` | pass | pass | pass | pass |
| `chk_513_1_8` | pass | pass | pass | pass |

#### guard_src1

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `q2k_e8_last1_b0` | **abort 2/2** | **abort 2/2** | **abort 2/2** | pass |
| `q2k_e8_last1_b1` | **abort 2/2** | **abort 2/2** | **abort 2/2** | pass |
| `q2k_e8_unif_b0` | **abort 2/2** | pass | pass | pass |
| `q2k_e128_last1_b1` | **abort 2/2** | **abort 2/2** | **abort 2/2** | pass |
| `q2k_e128_unif_b1` | **abort 2/2** | pass | pass | pass |
| `q4k_e128_u6_n5_b1` | **abort 2/2** | **abort 2/2** | pass | pass |
| `chk_65_8_8` | **abort 2/2** | pass | pass | pass |
| `chk_513_1_8` | pass | pass | pass | pass |

#### guard_ids_dst

| case | b11081 as pinned (`ne11`) | main's 903 (`ne12*n_used`) | 903 widest tile | 903 amended (widest padded) |
|---|---|---|---|---|
| `q2k_e8_last1_b0` | **abort 2/2** | **abort 2/2** | pass | pass |
| `q2k_e8_last1_b1` | **abort 2/2** | **abort 2/2** | pass | pass |
| `q2k_e8_unif_b0` | **abort 2/2** | **abort 2/2** | pass | pass |
| `q2k_e128_last1_b1` | **abort 2/2** | **abort 2/2** | pass | pass |
| `q2k_e128_unif_b1` | **abort 2/2** | **abort 2/2** | pass | pass |
| `q4k_e128_u6_n5_b1` | **abort 2/2** | **abort 2/2** | pass | pass |
| `chk_65_8_8` | **abort 2/2** | **abort 2/2** | pass | pass |
| `chk_513_1_8` | pass | pass | pass | pass |

#### what the device reported (MMQ_DEBUG_PRINT, debug build)

q2k_e8_last1_b0        none     MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_last1_b0        main     MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_last1_b0        widest   MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 80 covered
q2k_e8_last1_b0        amended  MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 85 covered
q2k_e8_last1_b1        none     MMQ_ID q2_K fb=0 ne11= 1 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_last1_b1        main     MMQ_ID q2_K fb=0 ne11= 1 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_last1_b1        widest   MMQ_ID q2_K fb=0 ne11= 1 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 80 covered
q2k_e8_last1_b1        amended  MMQ_ID q2_K fb=0 ne11= 1 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 85 covered
q2k_e8_unif_b0         none     MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_unif_b0         main     MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 0 SHORT 
q2k_e8_unif_b0         widest   MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 518 need= 79 pad= 80 covered
q2k_e8_unif_b0         amended  MMQ_ID q2_K fb=0 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 85 covered
q2k_e128_last1_b1      none     MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 9600 need= 79 pad= 0 SHORT 
q2k_e128_last1_b1      main     MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 9600 need= 79 pad= 0 SHORT 
q2k_e128_last1_b1      widest   MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 9600 need= 79 pad= 80 covered
q2k_e128_last1_b1      amended  MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 9600 need= 79 pad= 85 covered
q2k_e128_unif_b1       none     MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 9600 need= 79 pad= 0 SHORT 
q2k_e128_unif_b1       main     MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 9600 need= 79 pad= 0 SHORT 
q2k_e128_unif_b1       widest   MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 9600 need= 79 pad= 80 covered
q2k_e128_unif_b1       amended  MMQ_ID q2_K fb=0 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 9600 need= 79 pad= 85 covered
q4k_e128_u6_n5_b1      none     MMQ_ID q4_K fb=0 ne11= 1 ne12= 5 n_used= 6 ne02= 128 | J= 16 nthreads=128 | src1 need= 17 pad= 0 SHORT | ids_dst n= 30 need= 15 pad= 0 SHORT 
q4k_e128_u6_n5_b1      main     MMQ_ID q4_K fb=0 ne11= 1 ne12= 5 n_used= 6 ne02= 128 | J= 16 nthreads=128 | src1 need= 17 pad= 16 SHORT | ids_dst n= 30 need= 15 pad= 0 SHORT 
q4k_e128_u6_n5_b1      widest   MMQ_ID q4_K fb=0 ne11= 1 ne12= 5 n_used= 6 ne02= 128 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 30 need= 15 pad=128 covered
q4k_e128_u6_n5_b1      amended  MMQ_ID q4_K fb=0 ne11= 1 ne12= 5 n_used= 6 ne02= 128 | J= 16 nthreads=128 | src1 need= 17 pad=128 covered | ids_dst n= 30 need= 15 pad=128 covered
chk_65_8_8             none     MMQ_ID q2_K fb=0 ne11= 8 ne12= 65 n_used= 8 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 0 SHORT | ids_dst n= 520 need= 79 pad= 0 SHORT 
chk_65_8_8             main     MMQ_ID q2_K fb=0 ne11= 8 ne12= 65 n_used= 8 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 520 need= 79 pad= 0 SHORT 
chk_65_8_8             widest   MMQ_ID q2_K fb=0 ne11= 8 ne12= 65 n_used= 8 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 80 SHORT | ids_dst n= 520 need= 79 pad= 80 covered
chk_65_8_8             amended  MMQ_ID q2_K fb=0 ne11= 8 ne12= 65 n_used= 8 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 520 need= 79 pad= 85 covered
