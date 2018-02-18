#include <omp.h>

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <Windows.h>
#include <cstdint>

#include "immintrin.h"
#include "assert.h"
#define SIZE 1024


int main() {
	int THR = 4;
	int i, j, k;

	uint64_t start, end;
	double time_s, time_f;

	omp_set_dynamic(0);
	omp_set_num_threads(THR);

	register double(*__restrict A)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	register double(*__restrict B)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	register double(*__restrict C)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	register double(*__restrict C2)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));
	register double(*__restrict C3)[SIZE] = (double(*)[SIZE])malloc(SIZE * SIZE * sizeof(double));

	srand((unsigned int)time(NULL));

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			A[i][j] = rand() % 5;
			B[i][j] = rand() % 5;
		}
	}

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			C[i][j] = 0;			
		}
	}

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			C3[i][j] = 0;
		}
	}

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			C2[i][j] = 0;
		}
	}

	//	start = GetTickCount64();
	//
	//	double sum;
	//
	//#pragma omp parallel for
	//	for (k = 0; k < SIZE; k++) {
	//		for (j = 0; j < SIZE; j++) {
	//			for (i = 0; i < SIZE; i++) {
	//				C[j][i] += A[k][i] * B[j][k];
	//			}
	//		}
	//	}
	//
	//	end = GetTickCount64();
	//	printf("time not vect %f\n", (float)(end - start));
	//

	//time_s = omp_get_wtime();

	//start = GetTickCount64();
//#pragma omp parallel for
//	for (int i = 0; i < SIZE; i++) {
//		double *temp = C[i];
//		for (int j = 0; j < SIZE; j++) {
//			int temp1 = A[i][j];
//			double *temp2 = B[j];
//			for (int k = 0; k < SIZE; k++) {
//				temp[k] += temp1 * temp2[k];
//			}
//		}
//	}

	time_s = omp_get_wtime();
//#pragma omp parallel for
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				C[j][i] += A[j][k] * B[k][i];
			}
		}
	}
	time_f = omp_get_wtime();
	printf("Calculation time : %f \n",
		(time_f - time_s));


	time_s = omp_get_wtime();
#pragma omp parallel for 
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			for (k = 0; k < SIZE; k++) {
				C2[j][i] += A[j][k] * B[k][i];
			}
		}
	}
	time_f = omp_get_wtime();
	printf("Calculation time omp: %f \n",
		(time_f - time_s));
	//end = GetTickCount64();
	//printf("time vect %f\n", (float)(end - start));


	const int BlockSize = 256;
	const int GridSize = int(SIZE / double(BlockSize));
	const int BlockSizeMedium = 64;
	const int GridSizeMedium = int(BlockSize / double(BlockSizeMedium));
	const int BlockSizeSmall = 32;
	const int GridSizeSmall = int(BlockSizeMedium / double(BlockSizeSmall));

//
//	time_s = omp_get_wtime();
//#pragma omp parallel for 


	//for (int n = 0; n < GridSize; n++)
	//	for (int m = 0; m < GridSize; m++)
	//		for (int iter = 0; iter < GridSize; iter++)
	//			for (int i = n * BlockSize; i < (n + 1)*BlockSize; i++)
	//				for (int j = m * BlockSize; j < (m + 1)*BlockSize; j++)
	//					for (int k = iter * BlockSize; k < (iter + 1)*BlockSize; k++) {
	//						C3[j][i] += A[j][k] * B[k][i];
	//					}

	//						




//	time_f = omp_get_wtime();
//	printf("Calculation time K3+omp: %f \n",
//		(time_f - time_s));
//
//	time_s = omp_get_wtime();
//
//#pragma omp parallel for
	//for (int n = 0; n < GridSize; n++) {
	//	for (int m = 0; m < GridSize; m++) {

	//		for (int nM = 0; nM < GridSizeMedium; nM++)
	//		{
	//			for (int mM = 0; mM < GridSizeMedium; mM++) {
	//				for (int iterM = 0; iterM < GridSize*GridSizeMedium; iterM++) {

	//					for (int i = n * BlockSize + nM * BlockSizeMedium; i < n * BlockSize + (nM + 1)*BlockSizeMedium; i++)
	//					{
	//						for (int j = m * BlockSize + mM * BlockSizeMedium; j < m * BlockSize + (mM + 1)*BlockSizeMedium; j++)
	//						{								
	//							for (int k = iterM * BlockSizeMedium; k < (iterM + 1)*BlockSizeMedium; k++)
	//							{
	//								C3[j][i] += A[j][k] * B[k][i];
	//							}
	//						}

	//					}
	//				}
	//			}
	//		}		
	//	}
	//}
//	time_f = omp_get_wtime();
//	printf("Calculation time K3+K2+omp: %f \n",
//		(time_f - time_s));
//
	 
	time_s = omp_get_wtime();

#pragma omp parallel for
	for (int n = 0; n < GridSize; n++) {
		for (int m = 0; m < GridSize; m++) {

			for (int nM = 0; nM < GridSizeMedium; nM++){
				for (int mM = 0; mM < GridSizeMedium; mM++) {

					for (int nS = 0; nS < GridSizeSmall; nS++)	
						for (int mS = 0; mS < GridSizeSmall; mS++) {

							for (int iterS = 0; iterS <GridSize*GridSizeMedium*GridSizeSmall; iterS++) {
								for (int i = n * BlockSize + nM * BlockSizeMedium + nS * BlockSizeSmall; 
									i < n * BlockSize + (nS+1) * BlockSizeSmall + nM*BlockSizeMedium; i++)
								{
									for (int j = m * BlockSize + mM * BlockSizeMedium+ mS * BlockSizeSmall; 
										j < m * BlockSize + (mS + 1) * BlockSizeSmall + mM * BlockSizeMedium; j++)
									{										
										for (int k = iterS* BlockSizeSmall; k < (iterS + 1)*BlockSizeSmall; k++)
										{
											C3[j][i] += A[j][k] * B[k][i];
										}
									}

								}								
							}
						}					
				}
			}	
		}
	}
	time_f = omp_get_wtime();
	printf("Calculation time K3+K2+K1+omp: %f \n",
		(time_f - time_s));

	int k1 = 0, k2 = 0;
	//for (int i = 0; i < SIZE; i++) {
	//	for (int j = 0; j < SIZE; j++) {
	//		printf("%5.3f ", C[i][j]);
	//	}
	//	printf("\n");
	//}
	//printf("\n");
	//
	//for (int i = 0; i < SIZE; i++) {
	//	for (int j = 0; j < SIZE; j++) {
	//		printf("%5.3f ", C3[i][j]);
	//	}
	//	printf("\n");
	//}


	
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (C[i][j] != C2[i][j]) {
				k1 = 1;	
			}
		}
	}

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			if (C[i][j] != C3[i][j]) {
				k2 = 1;
				
			}
		}
	}

	if (k1 == 1) {
		printf("not omp!=omp \n");
	}
	if (k2 == 1) {
		printf("not omp!=K1+K2+K3+omp\n");
	}

	
	/*end = GetTickCount64();
	printf("time k opm %f\n", (float)(end - start));*/

	free(A);
	free(B);
	free(C);
	free(C2);
	free(C3);
	system("pause");
	return 0;
}