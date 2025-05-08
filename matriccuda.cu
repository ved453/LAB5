#include<iostream>
#include "cuda_runtime.h"
#include<chrono>
#include<vector>

using namespace std;


void multipyCPU(vector<float> &C, vector<float>& B, vector<float>&A, int N){
    size_t size = static_cast<size_t> (N)*N;
    if(A.size() != size || B.size() != size){
        cout << "Input matrix size mismatch ."<<std::endl;
        return;
    }

    if(C.size() != size){
        C.resize(size);
    }

    for(int row=0; row < N; ++row){
        for(int col =0; col< N; ++col){
            float sum = 0.0f;
            for(int k=0; k < N; k++)
            {
                sum += A[static_cast<size_t>(row)*N+k]*B[static_cast<size_t>(k)*N+col];
            }

            C[static_cast<size_t>(row)*N+col] = sum;
        }
    }
}

__global__
void multiplyGPU(float *A, float *B, float *C, int N){
    int row = blockIdx.x* blockDim.x + threadIdx.x;
    int col = blockIdx.y* blockDim.y + threadIdx.y;

    if(row < N && col < N){
        float cvalue = 0.0f;
        for(int k = 0; k< N; k++){
            cvalue += A[row*k+col]*B[k*N+col];
        }

        C[row*N+col] = cvalue;
    }
}


void intializeMatrix(vector<float> &A, int N){
    for(int i= 0; i < N; i++){
        A[i] = rand()%2000;
    }
}

int main(){


    int N;
     cout << "ENter N :";
     cin >> N;

    const int TILE_WIDTH = 16;
    size_t numElements = static_cast<size_t>(N)*N;
    size_t memsize = numElements * sizeof(float);


    vector<float> h_A(numElements);
    vector<float> h_B(numElements);
    vector<float> h_C_cpu(numElements,0.0f);
    vector<float> h_C_gpu(numElements,0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, memsize);
    cudaMalloc(&d_B, memsize);
    cudaMalloc(&d_C, memsize);


    intializeMatrix(h_A, N);
    intializeMatrix(h_B,N);

    cout << "\nPerforming Matrix Multiplication on CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    multipyCPU(h_C_cpu, h_A, h_B, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms" << std::endl;


    cudaMemcpy(d_A, h_A.data(), memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), memsize, cudaMemcpyHostToDevice);


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    int gridX = (N* dimBlock.x -1) / dimBlock.x;
    int gridY = (N* dimBlock.y -1) / dimBlock.y;

    dim3 dimGrid(gridX, gridY);


    std::cout << "Launching GPU Kernel and Synchronizing..." << std::endl;
    auto start_gpu_chrono = std::chrono::high_resolution_clock::now();


    multiplyGPU<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N);

    cudaDeviceSynchronize();

    auto end_gpu_chrono = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> gpu_duration_chrono = end_gpu_chrono - start_gpu_chrono;
    std::cout << "GPU Execution Time (measured with std::chrono + sync): "
              << gpu_duration_chrono.count() << " ms" << std::endl;


              cout << "------------SPEEDUP FACTOR-------- : "<< cpu_duration/ gpu_duration_chrono;


    cudaFree(d_A);
    cudaFree(d_B);
cudaFree(d_C);
              cudaDeviceReset();
              std::cout << "Cleanup finished. Exiting." << std::endl;
              return EXIT_SUCCESS;

    return 0;
}
