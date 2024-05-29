// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include<sys/time.h>


//Utilities
#include "includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "includes/helper_cuda.h"  
#include "includes/helper_functions.h"

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}

//Bigger size matrix
#define N 3 // 2^15 < 46340 < 2^16




int main(int argc, char** argv){

	float r[N] = {
		-0.04392401,
		0.08498386,
		-0.04105961
	};

	float alpha = 22.483807;

	float q[N] = {
		-0.001953309,
    	0.003780025,
    	-0.001825981
	};

	//float r1_d = nullptr;
	float *r_d = nullptr;
	float *q_d = nullptr;
	float ngtAlpha = -alpha;

	//(1)Allocate memory
	CHECK(cudaMalloc((void**)&r_d, N * sizeof(float)));
	//CHECK(cudaMalloc((void**)&r1_d, N * sizeof(float)));
	CHECK(cudaMalloc((void**)&q_d, N * sizeof(float)));

	//(2) Copy value from host to device
	CHECK(cudaMemcpy(r_d, r, N * sizeof(float), cudaMemcpyHostToDevice));
	printf("\n\nr_0 = \n");
	print_vector(r_d, N);
	
	CHECK(cudaMemcpy(q_d, q, N * sizeof(float), cudaMemcpyHostToDevice));
	printf("\n\nq_d = \n");
	print_vector(q_d, N);

	//(3) Create cublas handleer
	cublasHandle_t cublasHandle = 0;
	cublasCreate(&cublasHandle);
	// cublasStatus_t cublasStatus;
	// cublasStatus = cublasCreate(&cublasHandle);
	// checkCudaErrors(cublasHandle);

	//(4) Compute r_1 <- r_0
	checkCudaErrors(cublasSaxpy(cublasHandle, N, &ngtAlpha, q_d, 1, r_d, 1));
	printf("\n\nr_1 = \n");
	print_vector(r_d, N);


	//(5) Free memory
	checkCudaErrors(cublasDestroy(cublasHandle));
	CHECK(cudaFree(r_d));
	CHECK(cudaFree(q_d));


	return 0;
}

