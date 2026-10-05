### stock, memory pool, no sanitizer

| case | ne12 (#29941, master) | #29953 | #29953 + amendment |
|---|---|---|---|
| `ids16` | 3/3 | 3/3 | 3/3 |
| `dense321` | 3/3 | 3/3 | 3/3 |

### guard page on ids_dst only, no sanitizer

| case | ne12 (#29941, master) | #29953 | #29953 + amendment |
|---|---|---|---|
| `ids16` | 0/3 abort | 0/3 abort | 3/3 |
| `dense321` | 3/3 | 3/3 | 3/3 |

### guard page on src1 only, no sanitizer

| case | ne12 (#29941, master) | #29953 | #29953 + amendment |
|---|---|---|---|
| `ids16` | 0/3 abort | 0/3 abort | 3/3 |
| `dense321` | 3/3 | 0/3 abort | 3/3 |

