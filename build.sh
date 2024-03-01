#! /sbin/sh

nvcc --ptxas-options=-v --compiler-options '-fPIC' -o culib/deviceQuery.so --shared culib/deviceQuery.cpp
go run .