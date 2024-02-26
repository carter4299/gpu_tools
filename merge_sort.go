package main

/*
void merge_sort(unsigned int* arr, unsigned int N, unsigned int numValues);
#cgo LDFLAGS: -L. -lmerge_sort
*/
import "C"

import (
	"math/rand"
	"time"
	"unsafe"
)

const MaxInt32  = 1<<31 - 1
const MinInt32  = -1 << 31

const huh = 65536

// generateUInts generates a slice of N random uint values.
func generateUInts(N int) []int {
    var nums []int

    // Seed the random number generator to ensure different results on each run
	rand.New(rand.NewSource(time.Now().UnixNano()))

    for i := 0; i < N; i++ {
        // Convert to uint to fit the function's purpose. Adjust range as needed.
        nums = append(nums, int(rand.Intn(MaxInt32)))
    }

    return nums
}


func MergeSort(nums []uintptr, N uintptr) {
	// Convert the Go slice to a C array
	cNums := (*C.uint)(unsafe.Pointer(&nums[0]))
	cNumsSlice := (*[1 << 30]C.uint)(unsafe.Pointer(cNums))[:len(nums):len(nums)] // Convert C array to Go slice
	for i := 0; i < len(nums); i++ {
		cNumsSlice[i] = C.uint(nums[i])
	}


	// Call the C function
	C.merge_sort(cNums, C.uint(N), C.uint(huh))
}

func main() {
    N := 4 * 1048576
    nums := generateUInts(N)

	// Convert the Go slice to a C array
	var numsPtrs []uintptr
	for _, num := range nums {
		numsPtrs = append(numsPtrs, uintptr(num))
	}

	MergeSort(numsPtrs, uintptr(N))

}

//nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libmerge_sort.so --shared merge_sort.cu
//LD_LIBRARY_PATH=${PWD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} go run merge_sort.go

/*
[cartersusi@clinux gpu_sort]$ LD_LIBRARY_PATH=${PWD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} go run merge_sort.go
# command-line-arguments
./merge_sort.go:36:2: could not determine kind of name for C.merge_sort
cgo: 
gcc errors for preamble:
./merge_sort.go:4:17: error: unknown type name 'uint'; did you mean 'int'?
    4 | void merge_sort(uint* arr, uint N);
      |                 ^~~~
      |                 int
./merge_sort.go:4:28: error: unknown type name 'uint'; did you mean 'int'?
    4 | void merge_sort(uint* arr, uint N);
      |                            ^~~~
      |                            int

[cartersusi@clinux gpu_sort]$ 
*/