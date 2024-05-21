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

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}


//Hardcorded 3 by 3 matrix
#define N 3
float mtxA_h[N*N] = {
    1.5004, 1.3293, 0.8439,
    1.3293, 1.2436, 0.6936,
    0.8439, 0.6936, 1.2935
};

// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}




int main(int argc, char** argv)
{   
    // double startTime, endTime;
    //(1) Allocate device memory for arrays mtxA_d;
    float *mtxA_d = NULL;  //  Declare mtxA_d as a float pointer for device memoery
    CHECK(cudaMalloc((void**)&mtxA_d, sizeof(float) * (N * N))); //Allocate space in global memory
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, sizeof(float) * (N * N), cudaMemcpyHostToDevice)); // Copy date from host to device

    // Print the matrix from device to host (Check for Debugging)
    print_mtx(mtxA_d, N, (N * N));

    // Free the GPU memory after use
    cudaFree(mtxA_d);


} // end of main