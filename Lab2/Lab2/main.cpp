#include <omp.h>

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>
#include <cstdint>

#include "immintrin.h"

#define SIZE 512

//time not vect omp= ~110
//time vect opm = ~16
//time kash omp = ~31

int main() {
	int THR = 4;
	int i, j, k;
	
	uint64_t start, end;

	omp_set_dynamic(0);
	omp_set_num_threads(THR);

	int(*__restrict A)[SIZE] = (int(*)[SIZE])malloc(SIZE * SIZE * sizeof(int));
	int(*__restrict B)[SIZE] = (int(*)[SIZE])malloc(SIZE * SIZE * sizeof(int));
	int(*__restrict C)[SIZE] = (int(*)[SIZE])malloc(SIZE * SIZE * sizeof(int));

	srand((unsigned int)time(NULL));
	for (int i = 0; i<SIZE; i++) {
		for (int j = 0; j<SIZE; j++) {
			A[i][j] = rand() % 5;
			B[i][j] = rand() % 5;
		}
	}
	start = GetTickCount64();

#pragma omp parallel for
	for (k = 0; k < SIZE; k++) {
		for (j = 0; j < SIZE; j++) {
			for (i = 0; i < SIZE; i++) {
				C[j][i] += A[k][i] * B[j][k];
			}
		}
	}

	end = GetTickCount64();
	printf("time not vect %f\n", (float)(end - start));

	start = GetTickCount64();
#pragma omp parallel for
	for (int i = 0; i < SIZE; i++) {
		int *temp = C[i];
		for (int j = 0; j < SIZE; j++) {
			int temp1 = A[i][j];
			int *temp2 = B[j];
			for (int k = 0; k < SIZE; k++) {
				temp[k] += temp1 * temp2[k];
			}
		}
	}

	end = GetTickCount64();
	printf("time vect %f\n", (float)(end - start));

	int blockSize = SIZE / 2;

	start = GetTickCount64();

#pragma omp parallel for										////////////////////////////////////
	for (int l = 0; l < SIZE; l += blockSize) {
		for (int m = 0; m < SIZE; m += blockSize) {
			for (int n = 0; n < SIZE; n += blockSize) {
				for (int i = l; i < l + blockSize; i++) {
					int *temp = C[i];
					for (int j = m; j < m + blockSize; j++) {
						int temp1 = A[i][j];
						int *temp2 = B[j];
						for (int k = n; k < n + blockSize; k++) {
							temp[k] += temp1 * temp2[k];
						}
					}
				}
			}
		}
	}

	end = GetTickCount64();
	printf("time k %f\n", (float)(end - start));

	free(A);
	free(B);
	free(C);
	system("pause");
	return 0;
}