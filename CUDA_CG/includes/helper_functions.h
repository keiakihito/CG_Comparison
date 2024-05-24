#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include<sys/time.h>

// helper function CUDA error checking and initialization
#include "helper_cuda.h"  

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}




//Generate random SPD dense matrix
// N is matrix size
float* generateSPD_DenseMatrix(int N){
	float* mtx_h = NULL;
	float* mtx_d = NULL;
	float* mtxSPD_h = NULL;
	float* mtxSPD_d = NULL;

	//Using for cublas function
	const float alpha = 1.0f;
	const float beta = 0.0f;

	//Allocate memoery in Host
	mtx_h = (float*)calloc(N*N, sizeof(double));
	mtxSPD_h = (float*)calloc(N*N, sizeof(double));

	if(! mtx_h || ! mtxSPD_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h in all elements random values between 0 and 1.
	for (int wkr = 0; wkr < N*N;  wkr++){
		if(wkr % N == 0){
			printf("\n");
		}
		double randomVal = static_cast<double>(rand()/RAND_MAX);
		mtx_h[wkr] = round(randomVal *1e4 / 1e4);
		printf("\nmtx_h[%d] = %f", wkr, mtx_h[wkr]);

	}

	// // Generate and store to mtx_h tridiagonal random values between 0 and 1.
	// mtx_h[0] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// mtx_h[1] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// for (int wkr = 1; wkr < N -1 ;  wkr++){
	// 	mtx_h[(wkr * N) + (wkr-1)] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// 	mtx_h[wkr * N + wkr] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// 	mtx_h[(wkr * N) + (wkr+1)] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// }
	// mtx_h[(N*N)-2] = static_cast<float>(rand()) / RAND_MAX * 1e-6;
	// mtx_h[(N*N)-1] = static_cast<float>(rand()) / RAND_MAX * 1e-6;


	//(1)Allocate memoery in device
	CHECK(cudaMalloc((void**)&mtx_d, sizeof(double) * (N*N)));
	CHECK(cudaMalloc((void**)&mtxSPD_d, sizeof(double) * (N*N)));

	//(2) Copy value from host to device
	CHECK(cudaMemcpy(mtx_d, mtx_h, sizeof(double)* (N*N), cudaMemcpyHostToDevice));

	//(3) Calculate SPD matrix <- A' * A
	// Create a cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, mtx_d, N, mtx_d, N, &beta, mtxSPD_d, N));

	//(4) Copy value from device to host
	CHECK(cudaMemcpy(mtxSPD_h, mtxSPD_d, sizeof(double) * (N*N), cudaMemcpyDeviceToHost));
	
	//(5) Free memeory
	cudaFree(mtx_d);
	cudaFree(mtxSPD_d);
	cublasDestroy(handle);
	free(mtx_h);


	return mtxSPD_h;
} // enf of generateSPD_DenseMatrix

void validate(const float *mtxA_h, const float* x_h, float* rhs, int N){
    float rsum, diff, error = 0.0f;

    for (int rw_wkr = 0; rw_wkr < N; rw_wkr++){
        rsum = 0.0f;
        for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
            rsum += mtxA_h[rw_wkr*N + clm_wkr]* x_h[clm_wkr];
            // printf("\nrsum = %f", rsum);
        } // end of inner loop
        diff = fabs(rsum - rhs[rw_wkr]);
        if(diff > error){
            error = diff;
        }
        
    }// end of outer loop
    
    printf("\n\nTest Summary: Error amount = %f\n", error);

}// end of validate

#endif // HELPER_FUNCTIONS_H