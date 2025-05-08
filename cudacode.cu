#include <iostream>
#include <vector>
#include <chrono> 
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include <string> 
#include <algorithm> 

#include <cuda_runtime.h>

using namespace std;

inline void checkCudaErrors(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at " << file << ":" << line
                  << " code=" << result << " (\"" << cudaGetErrorString(result)
                  << "\")" << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(val) checkCudaErrors((val), __FILE__, __LINE__)


void initializeMatrix(std::vector<float>& mat, int N) {
    if (mat.size() != (size_t)N * N) {
        mat.resize((size_t)N * N);
    }
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

void printMatrix(const std::vector<float>& mat, int N, const std::string& title) {
    std::cout << "\n" << title << " (Dimensions: " << N << " x " << N << "):\n";
    int print_size = std::min(N, 16);
    if (N > 16) {
        std::cout << " (Showing top-left " << print_size << "x" << print_size << " corner)\n";
    }
    std::cout << "-----------------------------------------\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < print_size; ++i) {
        for (int j = 0; j < print_size; ++j) {
            std::cout << std::setw(8) << mat[static_cast<size_t>(i) * N + j] << " ";
        }
        std::cout << "\n";
    }
     if (print_size < N) std::cout << "...\n";
    std::cout << "-----------------------------------------\n";
}

bool compareMatrices(const std::vector<float>& mat1, const std::vector<float>& mat2, int N, float epsilon = 1e-5f) {
    if (mat1.size() != mat2.size() || mat1.size() != (size_t)N * N) {
        std::cerr << "Error: Matrix dimension mismatch in comparison." << std::endl;
        return false;
    }
    bool match = true;
    for (size_t i = 0; i < mat1.size(); ++i) {
        if (std::abs(mat1[i] - mat2[i]) > epsilon) {
            std::cerr << "Mismatch found at index " << i << " (row " << i / N << ", col " << i % N
                      << "): mat1=" << mat1[i] << ", mat2=" << mat2[i]
                      << ", diff=" << std::abs(mat1[i] - mat2[i]) << std::endl;
            match = false;
            return false;
        }
    }
    return match;
}


void matrixMulCPU(std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& B, int N) {
    size_t size = static_cast<size_t>(N) * N;
    if (A.size() != size || B.size() != size) {
         std::cerr << "Input matrix size mismatch." << std::endl;
         return;
    }
    if (C.size() != size) {
        C.resize(size);
    }

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[static_cast<size_t>(row) * N + k] * B[static_cast<size_t>(k) * N + col];
            }
            C[static_cast<size_t>(row) * N + col] = sum;
        }
    }
}


__global__ void matrixMulGPU(float *C, const float *A, const float *B, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float Cvalue = 0.0f;
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = Cvalue;
    }
}


int main() {
    int N;
    std::cout << "Enter the square matrix dimension (N x N): ";
    if (!(std::cin >> N) || N <= 0) {
        std::cerr << "Error: Invalid matrix dimension entered." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Using matrix dimensions: " << N << " x " << N << std::endl;

  
    const int TILE_WIDTH = 16;

   
    size_t numElements = static_cast<size_t>(N) * N;
    size_t memSize = numElements * sizeof(float);

   
    vector<float> h_A(numElements);
    vector<float> h_B(numElements);
    vector<float> h_C_cpu(numElements, 0.0f);
    vector<float> h_C_gpu(numElements, 0.0f);
    cout << "Host memory allocated successfully via std::vector.\n";

  
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, memSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, memSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, memSize));
    cout << "Device memory allocated successfully.\n";

    
    srand(static_cast<unsigned int>(std::time(nullptr)));
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    
    cout << "\nPerforming Matrix Multiplication on CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_C_cpu, h_A, h_B, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms" << std::endl;

    cout << "\nPerforming Matrix Multiplication on GPU..." << std::endl;


    std::cout << "Copying matrices A and B from Host to Device..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), memSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), memSize, cudaMemcpyHostToDevice));


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    int gridX = (N + dimBlock.x - 1) / dimBlock.x;
    int gridY = (N + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(gridX, gridY);
    std::cout << "GPU Grid Dimensions: (" << dimGrid.x << ", " << dimGrid.y
              << "), Block Dimensions: (" << dimBlock.x << ", " << dimBlock.y << ")" << std::endl;


    std::cout << "Launching GPU Kernel and Synchronizing..." << std::endl;
    auto start_gpu_chrono = std::chrono::high_resolution_clock::now();


    matrixMulGPU<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N);
    CHECK_CUDA_ERROR(cudaGetLastError());



    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto end_gpu_chrono = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> gpu_duration_chrono = end_gpu_chrono - start_gpu_chrono;
    std::cout << "GPU Execution Time (measured with std::chrono + sync): "
              << gpu_duration_chrono.count() << " ms" << std::endl;

   
    std::cout << "Copying result matrix C from Device to Host..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, memSize, cudaMemcpyDeviceToHost));

   
    std::cout << "\nVerifying results..." << std::endl;
    bool results_match = compareMatrices(h_C_cpu, h_C_gpu, N);
    if (results_match) {
        std::cout << "Result Verification: SUCCESS - CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Result Verification: FAILED - CPU and GPU results DO NOT match." << std::endl;
    }


    double gpu_time_ms = gpu_duration_chrono.count();
    if (gpu_time_ms > 0 && cpu_duration.count() > 0) {
        double speedup = cpu_duration.count() / gpu_time_ms;
        std::cout << "\nSpeedup (CPU Time / GPU Time): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    } else {
         std::cout << "\nCould not calculate speedup (CPU time=" << cpu_duration.count() << " ms, GPU time=" << gpu_time_ms << " ms)." << std::endl;
    }


    if (N <= 16) {
        printMatrix(h_C_cpu, N, "Resultant Matrix C (CPU)");
        printMatrix(h_C_gpu, N, "Resultant Matrix C (GPU)");
    } else {
        std::cout << "\nResult matrices too large to display full content (showing corners below)." << std::endl;
        printMatrix(h_C_cpu, N, "Resultant Matrix C of CPU ");
        printMatrix(h_C_gpu, N, "Resultant Matrix C of GPU");
    }


  
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    std::cout << "Cleanup finished. Exiting." << std::endl;
    return EXIT_SUCCESS;
}