### stock, memory pool, no sanitizer

| case | #29953 as published | #29953 head (`3070d927f`) |
|---|---|---|
| `ids16` | - | 3/3 |
| `dense321` | - | 3/3 |

### guard page on ids_dst only, no sanitizer

| case | #29953 as published | #29953 head (`3070d927f`) |
|---|---|---|
| `ids16` | 0/3 abort | 3/3 |
| `dense321` | 3/3 | 3/3 |

### guard page on src1 only, no sanitizer

| case | #29953 as published | #29953 head (`3070d927f`) |
|---|---|---|
| `ids16` | 0/3 abort | 3/3 |
| `dense321` | 0/3 abort | 3/3 |

