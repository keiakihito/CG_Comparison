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

	// float r[N] = {
	// 	-0.04392401,
	// 	0.08498386,
	// 	-0.04105961
	// };

	// float alpha = 22.483807;

	// float q[N] = {
	// 	-0.001953309,
    // 	0.003780025,
    // 	-0.001825981
	// };

	//Values in CUDA iteration
	float r[N] = {
		-0.043923,
		0.084984,
		-0.041059
	};

	
	float alpha = 22.483810;


	float q[N] = {
		-0.001952,
    	0.003781,
    	-0.001825
	};


	/*
	In iteration output from main.cu
	Iteration 3 before r_{i+1} <- r_{i} - alpha * q
	r = 
	-0.000041 
	-0.000037 
	-0.000030 
	*/

	/*
	This calculation result, which is same with MATLAB
	r_1 = 
	-0.000035 
	-0.000027 
	-0.000026 
	*/

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

	printf("\n\nAlpha = %f\n", alpha);
	printf("\n\nnegAlpha = %f\n", ngtAlpha);
	
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
	printf("\n\n");


	//(5) Free memory
	checkCudaErrors(cublasDestroy(cublasHandle));
	CHECK(cudaFree(r_d));
	CHECK(cudaFree(q_d));


	return 0;
}


/*
Sample Run
r_0 = 
-0.043923 
0.084984 
-0.041059 


Alpha = 22.483810


negAlpha = -22.483810


q_d = 
-0.001952 
0.003781 
-0.001825 


r_1 = 
-0.000035 
-0.000027 
-0.000026 

*/

