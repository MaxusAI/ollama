### stock, memory pool, no sanitizer

| case | #462: #29953 as merged + guard, b11351 | #462, src1 padded by J_best blocks (control) | #462, ids_dst unpadded (control) |
|---|---|---|---|
| `dense321` | 3/3 | - | - |
| `ids16` | 3/3 | - | - |

### exact-size cudaMalloc under memcheck

| case | #462: #29953 as merged + guard, b11351 | #462, src1 padded by J_best blocks (control) | #462, ids_dst unpadded (control) |
|---|---|---|---|
| `dense321` | 1/1 E=0 | - | - |
| `ids16` | 1/1 E=0 | - | - |

### guard page on ids_dst only, no sanitizer

| case | #462: #29953 as merged + guard, b11351 | #462, src1 padded by J_best blocks (control) | #462, ids_dst unpadded (control) |
|---|---|---|---|
| `dense321` | 3/3 | 3/3 | 3/3 |
| `ids16` | 3/3 | 3/3 | 0/3 abort |

### guard page on src1 only, no sanitizer

| case | #462: #29953 as merged + guard, b11351 | #462, src1 padded by J_best blocks (control) | #462, ids_dst unpadded (control) |
|---|---|---|---|
| `dense321` | 3/3 | 0/3 abort | 3/3 |
| `ids16` | 3/3 | 0/3 abort | 3/3 |

