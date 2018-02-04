#include <omp.h>

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>
#include <cstdint>

#include "immintrin.h"

#define SIZE 800

//time vect = ~460
//time opm = ~460

int main() {
	int THR = 8;
	int i, j, k;
	
	uint64_t start, end;

	omp_set_dynamic(0);
	omp_set_num_threads(THR);

	double(*__restrict A)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	double(*__restrict B)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	double(*__restrict C)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));

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
	printf("time %f\n", (float)(end - start));

	free(A);
	free(B);
	free(C);
	system("pause");
	return 0;
}