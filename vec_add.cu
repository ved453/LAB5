#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// CPU function for vector addition
void vectorAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Compare arrays
bool compareArrays(const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > 1e-5) return false;
    }
    return true;
}

int main() {
    int n;
    std::cout << "Enter vector size: ";
    std::cin >> n;

    size_t size = n * sizeof(float);


    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);

 
    srand(static_cast<unsigned>(time(NULL)));
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

  
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    double cpu_time = cpu_duration.count();


    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
    float gpu_time = gpu_time_ms / 1000.0f;

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);


    bool match = compareArrays(h_C_cpu, h_C_gpu, n);


    std::cout << "\nResults match: " << (match ? "Yes" : "No") << std::endl;
    std::cout << "CPU Time: " << cpu_time << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " seconds" << std::endl;
    if (gpu_time > 0) {
        std::cout << "Speedup (CPU/GPU): " << cpu_time / gpu_time << "x" << std::endl;
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
