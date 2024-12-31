# learn-with-us

learn Math, CUDA and Triton, from simple operation up to complex.

## how to CUDA

1. Make sure you have `nvcc`, after that,

```bash
nvcc cuda/1-add-vector-1d.cu -o 1-add-vector-1d-out
./1-add-vector-1d-out
```

2. You can debug using cuda-gdb,

```bash
nvcc -G -g -o test cuda/1-add-vector-1d.cu
cuda-gdb test
break vectorAdd
run
```
