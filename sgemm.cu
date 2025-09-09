#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "src/utils.cuh"

#include <cublas_v2.h>

#define cuda_check(err) (cuda_check(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s <kernel_id>\n", argv[0]);
        return -1;
    }

    int kernel_id = atoi(argv[1]);
    if(kernel_id < 0 || kernel_id > 2) {
        printf("Invalid kernel_id. Please choose between 0 and 2.\n");
        return -1;
    } 
        cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return -1;
    }

    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;

    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));

    random_matrix(A, M, K);
    random_matrix(B, K, N);
    random_matrix(C, M, N);

    copy_matrix(C, C_ref, M, N);

    float *d_A, *d_B, *d_C, *d_C_ref;

    cuda_check(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    cuda_check(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&d_C_ref, M * N * sizeof(float)));

    cuda_check(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_C_ref, C_ref, M * N * sizeof(float), cudaMemcpyHostToDevice));

    printf("Testing kernel ID: %d\n", kernel_id);
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    if(kernel_id != 0)
        test_kernel(0, d_A, d_B, d_C_ref, M, N, K, alpha, beta, handle); // Warm up CUBLAS
    test_kernel(kernel_id, d_A, d_B, d_C, M, N, K, alpha, beta, handle);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if(verify_matrix(C, C_ref, M, N)) {
        printf("Result verification: SUCCESS\n");
    } else {
        printf("Result verification: FAILED\n");
    }

    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < 10; i++) {
        test_kernel(kernel_id, d_A, d_B, d_C, M, N, K, alpha, beta, handle);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= 1000.0f; // Average time per run
    printf("Average execution time over 10 runs: %f ms\n", elapsed_time / 10.);
    printf("GFLOPS: %f\n", (2.0f * M * N * K) / (elapsed_time * 1e9f / 10.0f));
    fflush(stdout);

    copy_matrix(C_ref, C, M, N);


    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));
    cuda_check(cudaFree(d_C_ref));
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cublasDestroy(handle);
    return 0;
}