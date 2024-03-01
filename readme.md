# GPU Tools for Go [Development]
## Overview

This package is meant for Linux servers with Nvidia GPUs.\
It provides lighweight GPU algorithms for CPU intensive tasks at scale.
- Sorting ([]int, []int32, []int64, []float32, []float64)
- Sum ([]int, []int32, []int64, []float32, []float64)
- Min ([]int, []int32, []int64, []float32, []float64)
- Max ([]int, []int32, []int64, []float32, []float64)
- Avg ([]int, []int32, []int64, []float32, []float64)
- Filter, ([]int, []int32, []int64, []float32, []float64)(less than, greater than, eq, less eq than , greater eq than)


### Fully Functional

```go
type GPU struct

GpuInfo() (GPU, error)
PrintGPUInfo(GPU)
```

### Halfway Functional

```go
MergeSort(GPU, slice) //slice can be []int, []int32, []int64, []float32, []float64
RadixSort(GPU, slice) //slice can be []int, []int32, []int64, []float32, []float64
```

reduction, 
scan, 
filtering, 
and map-reduce patterns.