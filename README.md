# learn-cuda-with-us

learn CUDA with us, from simple operation up to complex.

## how-to

1. Make sure you have `nvcc`, after that,

```bash
nvcc 1-add-vector-1d.cu -o 1-add-vector-1d-out
./1-add-vector-1d-out
```

2. You can debug using cuda-gdb,

```bash
nvcc -G -g -o test 1-add-vector-1d.cu
cuda-gdb test
break vectorAdd
run
```
