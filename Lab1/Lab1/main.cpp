#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>
#include <cstdint>

#include "immintrin.h"

#define SIZE 800
#define LITTLE_SIZE 8

int main() {
	double ****firstMatrix, ****secondMatrix, ****resultMatrix;

	firstMatrix = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		firstMatrix[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			firstMatrix[i][j] = (double**)_aligned_malloc(LITTLE_SIZE * sizeof(double*), 16);
			for (int k = 0; k < LITTLE_SIZE; k++) {
				firstMatrix[i][j][k] = (double*)_aligned_malloc(LITTLE_SIZE * sizeof(double), 16);
			}
		}
	}

	secondMatrix = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		secondMatrix[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			secondMatrix[i][j] = (double**)_aligned_malloc(LITTLE_SIZE * sizeof(double*), 16);
			for (int k = 0; k < LITTLE_SIZE; k++) {
				secondMatrix[i][j][k] = (double*)_aligned_malloc(LITTLE_SIZE * sizeof(double), 16);
			}
		}
	}

	resultMatrix = (double****)malloc(SIZE * sizeof(double***));
	for (int i = 0; i < SIZE; i++) {
		resultMatrix[i] = (double***)malloc(SIZE * sizeof(double**));
		for (int j = 0; j < SIZE; j++) {
			resultMatrix[i][j] = (double**)_aligned_malloc(LITTLE_SIZE * sizeof(double*), 16);
			for (int k = 0; k < LITTLE_SIZE; k++) {
				resultMatrix[i][j][k] = (double*)_aligned_malloc(LITTLE_SIZE * sizeof(double), 16);
			}
		}
	}

	srand(time(NULL));

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < LITTLE_SIZE; k++) {
				for (int m = 0; m < LITTLE_SIZE; m++) {
					firstMatrix[i][j][k][m] = rand() % 100;
					secondMatrix[i][j][k][m] = rand() % 100;
				}
			}
		}
	}

	uint64_t start, end;

	start = GetTickCount64();
	for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < LITTLE_SIZE; i++)
			{
				double *__restrict temp = resultMatrix[m][n][i];
				for (int j = 0; j < LITTLE_SIZE; j++)
				{
					double temp1 = firstMatrix[m][n][i][j];
					double *__restrict temp2 = secondMatrix[m][n][j];

#pragma loop(no_vector)					
					for (int k = 0; k < LITTLE_SIZE; k++)
					{
						temp[k] += temp1 * temp2[k];
					}

				}
			}
		}
	}
	end = GetTickCount64();
	printf("time v %f\n", (float)(end - start));

	start = GetTickCount64();
	for (int m = 0; m < SIZE; m++) {
		for (int n = 0; n < SIZE; n++) {
			for (int i = 0; i < LITTLE_SIZE; i++)
			{
				double *__restrict temp = resultMatrix[m][n][i];
				for (int j = 0; j < LITTLE_SIZE; j++)
				{
					double temp1 = firstMatrix[m][n][i][j];
					double *__restrict temp2 = secondMatrix[m][n][j];

					const __m256d buff = {
						temp1,
						temp1,
						temp1,
						temp1
					};

					__m256d res = _mm256_add_pd(_mm256_mul_pd(buff, *reinterpret_cast<__m256d*>(temp2 + 0)), *reinterpret_cast<__m256d*>(temp + 0));
					memcpy(temp + 0, &res, sizeof(res));
					__m256d res1 = _mm256_add_pd(_mm256_mul_pd(buff, *reinterpret_cast<__m256d*>(temp2 + 4)), *reinterpret_cast<__m256d*>(temp + 4));
					memcpy(temp + 4, &res1, sizeof(res1));

				}
			}
		}
	}
	end = GetTickCount64();
	printf("time sse %f\n", (float)(end - start));

}