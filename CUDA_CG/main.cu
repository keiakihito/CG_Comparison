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
  
    //Declare matrix A, solution vector x, given residual r
    //Float pointers for device memoery
    float *mtxA_d = NULL;
    float *x_d = NULL;  // solution vector x
    float *r_d = NULL; // Residual
    float alpha, beta, alphamns1 = 0.0;
     

    //(0) Set initial guess and given vector b, right hand side
    float *x = (float*)malloc(sizeof(float) * N);
    float *rhs = (float*)malloc(sizeof(float) * N);
    
    for (int i = 0; i < N; i++){
        x[i] = 0.0;
        rhs[i] = 1.0;
    }//end of for

    //(1) Allocate space in global memory
    CHECK(cudaMalloc((void**)&mtxA_d, sizeof(float) * (N * N))); 
    CHECK(cudaMalloc((void**)&x_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&r_d, sizeof(float) * N));


    //(2) Copy value from host to device 
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, sizeof(float) * (N * N), cudaMemcpyHostToDevice)); 
    //✅
    // // Print the matrix from device to host (Check for Debugging)
    // print_mtx(mtxA_d, N, (N * N));
    // CHECK(cudaMemcpy())

    // x_{0}
    CHECK(cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice)); 
    printf("\n\nx_{0}\n");
    print_vector(x_d, N);

    // rhs, r_d is b vector initial guess is all 1.
    // The vector b is used only getting r_{0}, r_{0} = b - Ax where Ax = 0 vector
    // Then keep updating residual r_{i+1} = r_{i} - alpha*Ad_{i}
    CHECK(cudaMemcpy(r_d, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
    printf("\n\nr_{0} AKA given vector b\n");
    print_vector(r_d, N);







    // Free the GPU memory after use
    free(x);
    free(rhs);
    cudaFree(mtxA_d);
    cudaFree(x_d);
    cudaFree(r_d);


} // end of main