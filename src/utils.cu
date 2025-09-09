#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void cudaDeviceInfo() {
    int deviceCount;
    cuda_check(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, i), __FILE__, __LINE__);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
    }
}

void random_matrix(float *mat, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void copy_matrix(float *src, float *dst, int rows, int cols) {
    
    for (int i = 0; i < rows * cols; ++i) {
        dst[i] = src[i];
    }
}
void print_matrix(const float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

bool verify_matrix(const float *mat1, const float *mat2, int rows, int cols) {
    const float epsilon = 1e-3;
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(mat1[i] - mat2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

void test_cublas(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
    float *d_A, *d_B;
    float *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cuda_check(cudaMalloc((void**)&d_A, size_A), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_B, size_B), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_C, size_C), __FILE__, __LINE__);

    cuda_check(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    cublasStatus_t stat;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       d_B, N,
                       d_A, K,
                       &beta,
                       d_C, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %d\n", stat);
        exit(EXIT_FAILURE);
    }

    cuda_check(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    cuda_check(cudaFree(d_A), __FILE__, __LINE__);
    cuda_check(cudaFree(d_B), __FILE__, __LINE__);
    cuda_check(cudaFree(d_C), __FILE__, __LINE__);
}

void test_mygemm_v1(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    float *d_A, *d_B;
    float *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cuda_check(cudaMalloc((void**)&d_A, size_A), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_B, size_B), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_C, size_C), __FILE__, __LINE__);

    cuda_check(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    mygemm_kernel1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cuda_check(cudaGetLastError(), __FILE__, __LINE__);
    cuda_check(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cuda_check(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    cuda_check(cudaFree(d_A), __FILE__, __LINE__);
    cuda_check(cudaFree(d_B), __FILE__, __LINE__);
    cuda_check(cudaFree(d_C), __FILE__, __LINE__);
}

void test_mygemm_v2(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    float *d_A, *d_B;
    float *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cuda_check(cudaMalloc((void**)&d_A, size_A), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_B, size_B), __FILE__, __LINE__);
    cuda_check(cudaMalloc((void**)&d_C, size_C), __FILE__, __LINE__);

    cuda_check(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cuda_check(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(N, blockDim.x), CEIL_DIV(M, blockDim.y));

    mygemm_kernel2<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cuda_check(cudaGetLastError(), __FILE__, __LINE__);
    cuda_check(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cuda_check(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    cuda_check(cudaFree(d_A), __FILE__, __LINE__);
    cuda_check(cudaFree(d_B), __FILE__, __LINE__);
    cuda_check(cudaFree(d_C), __FILE__, __LINE__);
}

void test_kernel(int kernel_id, const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle) {
    switch (kernel_id) {
        case 0:
            test_cublas(A, B, C, M, N, K, alpha, beta, handle);
            break;
        case 1:
            test_mygemm_v1(A, B, C, M, N, K, alpha, beta);
            break;
        case 2:
            test_mygemm_v2(A, B, C, M, N, K, alpha, beta);
            break;
        default:
            fprintf(stderr, "Invalid kernel ID: %d\n", kernel_id);
            exit(EXIT_FAILURE);
    }
}