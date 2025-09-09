#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template<const int BLOCK_SIZE>
__global__ void mygemm_kernel2(const float *A, const float *B, float *C,int M,int N,int K,float alpha,float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    constexpr int BK = BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float Ads[BM][BK];
    __shared__ float Bds[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float Csub = 0.0f;
    for(int k=0;k<K; k+=BK){
        
        Ads[ty][tx] = A[ty * K  + tx];
        Bds[ty][tx] = B[(ty) * N + tx];
        
        __syncthreads();

        A += BK;
        B += BK * N;
        for(int n=0;n<BK;n++){
            Csub += Ads[ty][n] * Bds[n][tx];
        }
        __syncthreads();
    }
    C[ty * N + tx] = alpha * Csub + beta * C[ty * N + tx];

}