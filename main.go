package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello, World!")
	gpu, err := GpuInfo()
	if err != nil {
		fmt.Println(err)
	}
	PrintGPUInfo(gpu)
}


