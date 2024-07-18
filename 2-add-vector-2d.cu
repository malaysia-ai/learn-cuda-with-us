#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int columns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int i = row * columns + col;
    c[i] = a[i] + b[i];
    
}

int main() {
    int rows = 1000;
    int columns = 1000;
    size_t size = rows * columns * sizeof(int);
    
    int *h_a, *h_b, *h_c;
    
    int *d_a, *d_b, *d_c;
    
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    
    for (int i = 0; i < rows * columns; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 16 * 16 = 256
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((columns + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, columns);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < rows * columns; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error: %d + %d != %d\n", h_a[i], h_b[i], h_c[i]);
            break;
        }
    }
    
    printf("Vector addition completed successfully.\n");
    
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}