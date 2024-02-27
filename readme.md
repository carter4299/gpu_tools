# **NOT FUNCTIONCAL**

## Needed
- Dynamic sizing 
```C
#define SHARED_SIZE_LIMIT 1024U
#define SAMPLE_STRIDE 128
static uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const uint MAX_SAMPLE_COUNT = 32768;


assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
assert(N % SHARED_SIZE_LIMIT == 0);
```
```go
//only sizing 1048576 * x works
N := 4 * 1048576
nums := generateUInts(N)
```

- type for int64(C.ulong), float32(C.float), float64(C.double)

