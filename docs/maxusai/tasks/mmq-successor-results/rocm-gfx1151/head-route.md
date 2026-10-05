
#### print

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `q2k_e8_last1_b0` | pass |
| `q2k_e8_last1_b1` | pass |
| `q2k_e8_unif_b0` | pass |
| `q2k_e128_last1_b1` | pass |
| `q2k_e128_unif_b1` | pass |
| `q4k_e128_u6_n5_b1` | pass |
| `chk_65_8_8` | pass |
| `chk_513_1_8` | pass |

#### guard_src1

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `q2k_e8_last1_b0` | pass |
| `q2k_e8_last1_b1` | pass |
| `q2k_e8_unif_b0` | pass |
| `q2k_e128_last1_b1` | pass |
| `q2k_e128_unif_b1` | pass |
| `q4k_e128_u6_n5_b1` | pass |
| `chk_65_8_8` | pass |
| `chk_513_1_8` | pass |

#### guard_ids_dst

| case | llama.cpp#29953 at 3070d927f |
|---|---|
| `q2k_e8_last1_b0` | pass |
| `q2k_e8_last1_b1` | pass |
| `q2k_e8_unif_b0` | pass |
| `q2k_e128_last1_b1` | pass |
| `q2k_e128_unif_b1` | pass |
| `q4k_e128_u6_n5_b1` | pass |
| `chk_65_8_8` | pass |
| `chk_513_1_8` | pass |

#### what the device reported (MMQ_DEBUG_PRINT, debug build)

q2k_e8_last1_b0        head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 79 covered
q2k_e8_last1_b1        head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 1 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 79 covered
q2k_e8_unif_b0         head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 7 ne12= 74 n_used= 7 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 518 need= 79 pad= 79 covered
q2k_e128_last1_b1      head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 9600 need= 79 pad= 79 covered
q2k_e128_unif_b1       head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 1 ne12= 600 n_used= 16 ne02= 128 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 9600 need= 79 pad= 79 covered
q4k_e128_u6_n5_b1      head     MMQ_IDS q4_K fb=0 prec=q8 ne11= 1 ne12= 5 n_used= 6 ne02= 128 | J= 16 nthreads=128 | src1 need= 17 pad= 17 covered | ids_dst n= 30 need= 15 pad= 15 covered
chk_65_8_8             head     MMQ_IDS q2_K fb=0 prec=q8 ne11= 8 ne12= 65 n_used= 8 ne02= 8 | J= 80 nthreads=256 | src1 need= 85 pad= 85 covered | ids_dst n= 520 need= 79 pad= 79 covered
chk_513_1_8            head     MMQ_DENSE q2_K fb=0 prec=q8 ne11= 74 ne12= 1 n_used= 0 ne02= 1 | J= 80 nthreads=256 | src1 need= 12 pad= 85 covered | no ids_dst
