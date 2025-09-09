#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void cuda_check(cudaError_t err, const char* file, int line);
void cudaDeviceInfo();

void random_matrix(float *mat, int rows, int cols);
void copy_matrix(float *src, float *dst, int rows, int cols);
void print_matrix(const float *mat, int rows, int cols);
bool verify_matrix(const float *mat1, const float *mat2, int rows, int cols);

float get_time();
float cpu_elapsed(float &t_start, float &t_end);

void test_kernel(int kernel_id, const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle);