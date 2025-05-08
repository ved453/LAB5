#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath> // Include cmath for fabs
using namespace std;
using namespace std::chrono;
#define N 10000000
#define M 30

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void matrixMul(float *A, float *B, float *C, int m) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < m) {
        float sum = 0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * m + col];
        }
        C[row * m + col] = sum;
    }
}

int main() {
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    vector<float> cpu_c_vector(N);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));

    auto start_cpu = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cpu_c_vector[i] = h_a[i] + h_b[i];
    }
    auto end_cpu = high_resolution_clock::now();
    double cpu_time = duration<double>(end_cpu - start_cpu).count();

    cudaEvent_t start, stop;
    float gpu_time;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    vectorAdd<<< (N + 255) / 256, 256 >>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "Vector Addition CPU time: " << cpu_time << " sec\n";
    cout << "Vector Addition GPU time: " << gpu_time / 1000.0 << " sec\n";
    cout << "Speedup: " << cpu_time / (gpu_time / 1000.0) << "\n";

    bool vector_match = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - cpu_c_vector[i]) > 1e-5) {
            vector_match = false;
            break;
        }
    }
    cout << "Vector Addition Results: " << (vector_match ? "Matched" : "Not Matched") << "\n\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[M * M];
    h_B = new float[M * M];
    h_C = new float[M * M];
    vector<float> cpu_C_vector(M * M);

    for (int i = 0; i < M * M; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, M * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * M * sizeof(float)));

    start_cpu = high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0;
            for (int k = 0; k < M; k++) {
                sum += h_A[i * M + k] * h_B[k * M + j];
            }
            cpu_C_vector[i * M + j] = sum;
        }
    }
    end_cpu = high_resolution_clock::now();
    cpu_time = duration<double>(end_cpu - start_cpu).count();

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, M * M * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M);
    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * M * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "Matrix Multiplication CPU time: " << cpu_time << " sec\n";
    cout << "Matrix Multiplication GPU time: " << gpu_time / 1000.0 << " sec\n";
    cout << "Speedup: " << cpu_time / (gpu_time / 1000.0) << "\n";

    bool matrix_match = true;
    for (int i = 0; i < M * M; i++) {
        if (fabs(h_C[i] - cpu_C_vector[i]) > 1e-5) {
            matrix_match = false;
            break;
        }
    }
    cout << "Matrix Multiplication Results: " << (matrix_match ? "Matched" : "Not Matched") << "\n";

    // Cleanup
    delete[] h_a, h_b, h_c, h_A, h_B, h_C;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// delete[] h_a;
// delete[] h_b;
// delete[] h_c;
// delete[] h_A;
// delete[] h_B;
// delete[] h_C;
