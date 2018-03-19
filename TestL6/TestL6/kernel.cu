
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <conio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <intrin.h>
#include <omp.h>

using namespace cv;
using namespace std;
const int nx = 32;
const int ny = 16;
const int N = 4;


__global__ void SumGPU(unsigned *a, char *b, unsigned char *r,
	int cols, int rows, int pitch, int pitch1) {

	__shared__ unsigned smem[ny + 1][nx + 1];
	unsigned tempBlock[12] = { 0,0,0,0,0,0,0,0,0,0,0,0 };

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;

	if (blockIdx.x == (gridDim.x - 1) || blockIdx.y == (gridDim.y - 1))
		return;

	if (xid < cols && yid < rows); {

		if ((threadIdx.y + 1) == ny && (yid + 1) < rows)
		{
			smem[threadIdx.y][threadIdx.x] = a[yid * pitch + xid];
			smem[threadIdx.y + 1][threadIdx.x] = a[(yid + 1) * pitch + xid];

			if ((threadIdx.x + 1) == nx)
			{
				smem[threadIdx.y][(threadIdx.x + 1)] = a[yid * pitch + xid + 1];
				smem[threadIdx.y + 1][threadIdx.x + 1] = a[(yid + 1) * pitch + xid + 1];
				//printf("CPU: %u__ GPU: %u", a[yid * pitch + xid * N + e], smem[threadIdx.y][threadIdx.x * N + e]);
			}
		}
		else
		{
			smem[threadIdx.y][threadIdx.x] = a[yid * pitch + xid];

			if ((threadIdx.x + 1) == nx)
			{
				smem[threadIdx.y][(threadIdx.x + 1)] = a[yid * pitch + xid + 1];
				//printf("CPU: %u__ GPU: %u", a[yid * pitch + xid * N + e], smem[threadIdx.y][threadIdx.x * N + e]);
			}
		}
		__syncthreads();


		unsigned char *ptr = (unsigned char *)smem;
		if (yid < rows - 1 && xid < cols - 1)
		{
			int esx = 0, esy = 0;
			if (threadIdx.x == 0) {
				esx = 1;
			}
			if (threadIdx.y == 0) {
				esy = 1;
			}
			for (; esx < 4 && threadIdx.x * N + esx < cols; esx++)
			{
				int basex = threadIdx.x * N - 1 + esx;
				int basey = threadIdx.y - 1 + esy;
				int sum = 0;
				for (int yt = basey; yt < basey + 3; yt++) {
					for (int xt = basex; xt < basex + 3; xt++) {
						sum += ptr[yt*(nx)*N + 1 + xt] * b[(yt - basey) * 3 + (xt - basex)];
					}
				}
				if (sum > 255) {
					sum = 255;
				}
				if (sum < 0) {
					sum = 0;
				}
				r[yid * pitch1 + xid * N + esx] = (unsigned char)sum;
			}
		}

		//unsigned char *ptr = (unsigned char *)smem;
		//if (yid < rows - 1 && xid*N < cols - 1) 
		//{
		//	int esx = 0, esy = 0;
		//	if (threadIdx.x == 0) {
		//		esx = 1;				
		//	}
		//	if (threadIdx.y == 0) {
		//		esy = 1;
		//	}
		//	for (; esx < 4 && threadIdx.x * N + esx < cols; esx++)
		//	{
		//		int basex = threadIdx.x * N - 1 + esx;
		//		int basey = threadIdx.y - 1 + esy;
		//		int sum = 0;
		//		for (int yt = basey; yt < basey + 3; yt++) {
		//			for (int xt = basex; xt < basex + 3; xt++) {
		//				sum += ptr[yt*(nx)*N + 1 + xt] * b[(yt - basey) * 3 + (xt - basex)];
		//			}
		//		}
		//		if (sum > 255) {
		//			sum = 255;
		//		}
		//		if (sum < 0) {
		//			sum = 0;
		//		}
		//		r[yid * pitch1 + xid*N + esx] = (unsigned char)sum;
		//	}			
		//}		
	}

}

int main()
{
	Mat img = imread("D:\\test.pgm", IMREAD_UNCHANGED);
	normalize(img, img, 0, 255, NORM_MINMAX);
	img.convertTo(img, CV_8U);

	Mat imgRes = img.clone();
	unsigned char *A = (unsigned char*)_aligned_malloc(img.rows * img.cols * sizeof(unsigned char), 8);
	char B[] = { -1,-1,-1,-1,9,-1,-1,-1,-1 };

	srand((unsigned int)time(NULL));
	double time_s, time_f;
	time_s = omp_get_wtime();

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			int basei = i - 1;
			int basej = j - 1;
			int sum = 0;
			for (int it = basei; it < basei + 3; it++) {
				for (int jt = basej; jt < basej + 3; jt++)
				{
					sum += img.data[it*img.cols + jt] * B[(it - basei) * 3 + (jt - basej)];
				}
			}

			if (sum > 255) {
				sum = 255;
			}
			if (sum < 0) {
				sum = 0;
			}
			imgRes.data[i*img.cols + j] = (unsigned char)sum;
		}
	}

	time_f = omp_get_wtime();
	printf("CPU: %f \n",
		(time_f - time_s));

	unsigned char *dev_R;
	unsigned *dev_A;
	char *dev_B;
	Mat imgResGPU = img.clone();

	size_t pitch, pitch1;
	size_t host_orig_pitch = img.cols * sizeof(unsigned char);
	size_t host_orig_pitch_A = img.cols * sizeof(unsigned);

	cudaError_t error = cudaMallocPitch((void**)&dev_A, &pitch, img.cols * sizeof(unsigned), img.rows);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMallocPitch((void**)&dev_R, &pitch1, img.cols * sizeof(unsigned char), img.rows);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy2D(dev_A, pitch, img.data, host_orig_pitch, img.cols * sizeof(unsigned char), img.rows, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy2D(dev_R, pitch1, img.data, host_orig_pitch, img.cols * sizeof(unsigned char), img.rows, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMalloc((void**)&dev_B, 9 * sizeof(char));
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(dev_B, B, 9 * sizeof(char), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}


	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 threads(nx, ny);

	int blockx = (img.cols + nx * N - 1) / (nx * N);  //pitch % nx*N == 0 ? pitch / (nx *N) : pitch / (nx*N) + 1;
	int blocky = (img.rows + ny - 1) / (ny);  //pitch % ny == 0 ? pitch / ny : pitch / ny + 1;
	dim3 blocks(blockx, blocky);

	time_s = omp_get_wtime();
	error = cudaEventSynchronize(start);
	SumGPU << <blocks, threads >> > (dev_A, dev_B, dev_R, img.cols, img.rows, pitch / sizeof(unsigned), pitch1);
	error = cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	time_f = omp_get_wtime();
	printf("GPU_1: %f \n",
		(time_f - time_s));

	error = cudaMemcpy2D(imgResGPU.data, host_orig_pitch, dev_R, pitch1, imgResGPU.cols, imgResGPU.rows, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}



	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_R);

	printf("OK\n");
	imwrite("D:\\outputGPU.pgm", imgResGPU);
	imwrite("D:\\outputCPU.pgm", imgRes);
	imwrite("D:\\outputImg.pgm", img);
	//imshow("img", img);
	//imshow("imgRes", imgRes);
	//imshow("imgResGPU", imgResGPU);	
	waitKey(0);
	return 0;
}
