#! /sbin/sh

nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libmerge_sort.so --shared merge_sort.cu
LD_LIBRARY_PATH=${PWD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} go run merge_sort.go