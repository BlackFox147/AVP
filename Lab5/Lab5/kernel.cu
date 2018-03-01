#include<cuda.h>
#include<cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <intrin.h>
#include <omp.h>

const int n = 32;
//int N = 32;


void SumCPU(int *A, int *B, int N) {

	unsigned __int64 begin, end;

	srand((unsigned int)time(NULL));
	double time_s, time_f;
	time_s = omp_get_wtime();
	for (int blockX = 0; blockX < N; blockX++) {
		for (int blockY = 0; blockY < N; blockY++) {


			for (int i = blockX * n; i < (blockX + 1)*n; i++) {
				for (int j = blockY * n; j < (blockY + 1)*n; j++) {

					for (int ki = blockX * n; ki <= i; ki++) {
						for (int kj = blockY * n; kj <= j; kj++) {
							int c = A[ki * n * N + kj];
							B[i*n *N + j] += A[ki * n *N + kj];
						}
					}
					B[i*n *N + j] -= A[i*n*N + j];

				}
			}

		}
	}

	time_f = omp_get_wtime();
	printf("CPU: %f \n",
		(time_f - time_s));

}

__global__ void SumGPU(int *a, int *b, int N) {
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int dx = blockDim.x;
	int dy = blockDim.y;
	int i = blockX * dx + tx;
	int j = blockY * dy + ty;

	for (int ki = blockX * n; ki <= i; ki++) {
		for (int kj = blockY * n; kj <= j; kj++) {
			int c = a[ki * n*N + kj];
			b[i*n *N + j] += a[ki * n*N + kj];
		}
	}
	b[i*n*N + j] -= a[i*n*N + j];
}



__global__ void SumGPU_2(int *a, int *b, int N) {
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int dx = blockDim.x;
	int dy = blockDim.y;
	int j = blockX * dx + tx;
	int i = blockY * dy + ty;

	for (int ki = blockY * n; ki <= i; ki++) {
		for (int kj = blockX * n; kj <= j; kj++) {
			int c = a[ki * n*N + kj];
			b[i*n *N + j] += a[ki * n*N + kj];
		}
	}
	b[i*n*N + j] -= a[i*n*N + j];
}



using namespace std;

void main(int argc, char* argv[])
{
	int count;
	printf("count = ");
	scanf("%d",&count);
	int N = count % n == 0 ? count / n : count / n + 1;

	double time_s, time_f;

	int *A = (int*)_aligned_malloc(n *n *N*N * sizeof(int), 32);
	int *B = (int*)_aligned_malloc(n *n *N*N * sizeof(int), 32);


	for (int i = 0; i < n*N; i++)
	{
		for (int j = 0; j < n*N; j++)
		{
			A[i*n*N + j] = i * n*N + j;
		}
	}

	memcpy(B, A, n *n *N*N * sizeof(int));

	SumCPU(A, B, N);

	//for (int i = 0; i < n*N; i++)
	//{
	//	for (int j = 0; j < n*N; j++)
	//	{
	//		printf("%u\t", A[i*n*N + j]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");
	//printf("\n");


	/*for (int i = 0; i < n*N; i++)
	{
	for (int j = 0; j < n*N; j++)
	{
	printf("%u\t", B[i*n*N + j]);
	}
	printf("\n");
	}

	printf("\n");
	printf("\n");*/

	int *dev_A, *dev_X, *dev_B, *dev_B2;

	int *a = (int*)_aligned_malloc(n *n *N*N * sizeof(int), 32);
	int *b = (int*)_aligned_malloc(n *n *N*N * sizeof(int), 32);
	int *b_2 = (int*)_aligned_malloc(n *n *N*N * sizeof(int), 32);

	memcpy(a, A, n *n *N*N * sizeof(int));
	memcpy(b, A, n *n *N*N * sizeof(int));
	memcpy(b_2, A, n *n *N*N * sizeof(int));

	cudaError_t error = cudaMalloc((void**)&dev_A, n *n *N*N * sizeof(int));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMalloc((void**)&dev_B, n *n *N*N * sizeof(int));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMalloc((void**)&dev_B2, n *n *N*N * sizeof(int));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(dev_A, a, n *n *N*N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(dev_B, b, n *n *N*N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(dev_B2, b_2, n *n *N*N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 threads(n, n);
	//dim3 blocks((n + (fiber - 1)) / fiber, (n + (fiber - 1)) / fiber);
	dim3 blocks(N, N);

	//begin = __rdtsc();
	time_s = omp_get_wtime();
	cudaEventSynchronize(start);
	SumGPU << <blocks, threads >> >(dev_A, dev_B, N);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//end = __rdtsc() - begin;

	//printf("runTimeGPU1 =  %llu \n", end - begin);
	time_f = omp_get_wtime();
	printf("GPU_1: %f \n",
		(time_f - time_s));

	error = cudaMemcpy(b, dev_B, n*n *N*N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	float timer = 0;

	time_s = omp_get_wtime();
	cudaEventSynchronize(start);
	SumGPU_2 << <blocks, threads >> >(dev_A, dev_B2, N);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//end = __rdtsc() - begin;

	//printf("runTimeGPU1 =  %llu \n", end - begin);
	time_f = omp_get_wtime();
	printf("GPU_2: %f \n",
		(time_f - time_s));

	error = cudaMemcpy(b_2, dev_B, n*n *N*N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	timer = 0;


	//cudaEventElapsedTime(&timer, start, stop);
	//printf("GPU = %f ms\n", timer);


	//cout << "runTimeGPU = " << timer << endl;

	/*for (int i = 0; i < n*N; i++)
	{
	for (int j = 0; j < n*N; j++)
	{
	printf("%u\t", b[i*n*N + j]);
	}
	printf("\n");
	}
	printf("\n");
	printf("\n");
	*/
	int ch = 0;
	for (int i = 0; i < n*N; i++)
	{
		for (int j = 0; j < n*N; j++)
		{
			if (B[i*n*N + j] != b[i*n*N + j]) {
				printf("ERROR!!! b1\n");
				ch = 1;
			}

			if (B[i*n*N + j] != b_2[i*n*N + j]) {
				printf("ERROR!!! b2\n");
				ch = 1;
			}

			if (ch == 1) {
				return;
			}
		}
	}

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_B2);

}
