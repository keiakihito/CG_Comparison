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
    float *x_d = NULL;  // Solution vector x
    float *r_d = NULL; // Residual
    float *dirc_d = NULL; // Direction
    float *Ax_d = NULL; // Vector Ax
    float *q_d = NULL; // Vector Ad
    float dot = 0.0f; // temporary val for d^{T} *q to get aplha
    
    //Using for cublas functin argument
    float alpha = 1.0; 
    float alphamns1 = -1.0;// negative alpha
    float beta = 0.0;

    float delta_new = 0.0;
    float delta_old = 0.0;
    const float EPS = 1e-5f;
    const int MAX_ITR = 10000;

    // In CG iteration alpha and beta
    float alph = 0.0f;
    float ngtAlph = 0.0f;
    float bta = 0.0f;
    
    //Stride for x and y using contiguous vectors
    //Using for calling cublasSgemv
    int strd_x = 1;
    int strd_y = 1;
     




    //(0) Set initial guess and given vector b, right hand side
    float *x = (float*)malloc(sizeof(float) * N);
    float *rhs = (float*)malloc(sizeof(float) * N);
    
    for (int i = 0; i < N; i++){
        x[i] = 0.0;
        rhs[i] = 1.0;
    }//end of for

    //(1) Allocate space in global memory
    CHECK(cudaMalloc((void**)&mtxA_d, sizeof(float) * (N * N)));
    // CHECK(cudaMalloc((void**)&x_h, sizeof(float) * N)); 
    CHECK(cudaMalloc((void**)&x_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&r_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&dirc_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&Ax_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&q_d, sizeof(float) * N));


    //(2) Copy value from host to device 
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, sizeof(float) * (N * N), cudaMemcpyHostToDevice)); 
    //✅
    // // Print the matrix from device to host (Check for Debugging)
    // print_mtx(mtxA_d, N, (N * N));
    // CHECK(cudaMemcpy())

    // x_{0}
    CHECK(cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice)); 
    //✅
    // printf("\n\nx_{0}\n");
    // print_vector(x_d, N);

    // rhs, r_d is b vector initial guess is all 1.
    // The vector b is used only getting r_{0}, r_{0} = b - Ax where Ax = 0 vector
    // Then keep updating residual r_{i+1} = r_{i} - alpha*Ad_{i}
    CHECK(cudaMemcpy(r_d, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
    //✅
    // printf("\n\nr_{0} AKA given vector b\n");
    // print_vector(r_d, N);


    //(3) Handle to the CUBLAS context
    //The cublasHandle variable will be used to store the handle to the cuBLAS library context. 
    cublasHandle_t cublasHandle = 0;
    //The status will help in error checking and ensuring that cuBLAS functions are executed successfully.
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    //This function checks the returned status (cublasStatus) from the cublasCreate call.
    //If the setting up resources fails, it ends program.
    checkCudaErrors(cublasStatus);



    //(5) Iteration
    /* 💫💫💫Begin CG💫💫💫 */
    //Setting up the initial state.
    /*
    1. Calculate Ax_{0}
    2. Find residual r_{0} = b - Ax{0}
    3. Set d <- r
    4. Set delta_{new} <- r^{T} * r
    */

    //1. Calculate Ax_{0}
    checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, x_d, strd_x, &beta, Ax_d, strd_y));
    //✅
    // printf("\n\n~~vector Ax_{0}~~\n");
    // print_vector(Ax_d, N);

    //2. Find residual r_{0} = b - Ax{0}
    //This function performs the operation y=αx+y.
    // Update the residual vector r by calculating r_{0} = b - Ax_{0}
    // Given vector b is d_r only used 1 time. We will updateing d_r as a new residual.
    // which is critical for determining the direction and magnitude of the initial search step in the CG algorithm.
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alphamns1, Ax_d, strd_x, r_d, strd_y));
    // //✅
    // printf("\n\n~~vector r_{0}~~\n");
    // print_vector(r_d, N);

    //3. Set d <- r;
    CHECK(cudaMemcpy(dirc_d, r_d, N * sizeof(float), cudaMemcpyDeviceToDevice));
    // //✅
    // printf("\n\n~~vector d_{0}~~\n");
    // print_vector(dirc_d, N);

    //4,  delta_{new} <- r^{T} * r
    // Compute the squared norm of the initial residual vector r (stored in r1).
    checkCudaErrors(cublasSdot(cublasHandle, N, r_d, strd_x, r_d, strd_y, &delta_new));
    // //✅
    // printf("\n\n~~vector delta_new{0}~~\n %f\n ", delta_new);
    
    int cntr = 1; // counter

    while(delta_new > EPS * EPS && cntr <= MAX_ITR){
        //q <- Ad
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, dirc_d, strd_x, &beta, q_d, strd_y));
        // //✅
        // printf("\n\n~~q_d AKA vector Ad~~\n");
        // print_vector(q_d, N);


        //dot <- d^{T} * q
        checkCudaErrors((cublasSdot(cublasHandle, N, dirc_d, strd_x, q_d, strd_y, &dot)));
        // //✅
        // printf("\n\n~~dot AKA (d^{T*q)~~\n %f\n", dot);

        //alpha(a) <- delta_{new} / dot // dot <- d^{T} * q 
        alph = delta_new / dot;
        // //✅
        // printf("\n\n~~alph ~~\n %f\n", alph);

        //x_{i+1} <- x_{i} + alpha * d_{i}
        checkCudaErrors((cublasSaxpy(cublasHandle, N, &alph, dirc_d, strd_x, x_d, strd_y)));
        // //✅
        // printf("\n\n~~x_{i+1}~~\n");
        // print_vector(x_d, N);






        if(cntr % 50 == 0){
            //r <- b -Ax Recompute

            //r_{0} <- b
            CHECK(cudaMemcpy(r_d, rhs, N * sizeof(float), cudaMemcpyHostToDevice));
            // //✅
            // printf("\n\n~~vector r_{0}~~\n");
            // print_vector(r_d, N);
            
            //Ax_d <- A * x
            checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, x_d, strd_x, &beta, Ax_d, strd_y));
            //✅
            // printf("\n\n~~vector Ax_{0}~~\n");
            // print_vector(Ax_d, N);

            //r_{0} = b- Ax
            checkCudaErrors(cublasSaxpy(cublasHandle, N, &alphamns1, Ax_d, strd_x, r_d, strd_y));
            //✅
            // printf("\n\n~~vector r_{0}~~\n");
            // print_vector(r_d, N);
        }else{
            // Set -alpha
            ngtAlph = -alph;

            //r_{i+1} <- r_{i} -alpha*q
            checkCudaErrors(cublasSaxpy(cublasHandle, N, &ngtAlph, q_d, strd_x, r_d, strd_y));
            //✅
            // printf("\n\n~~vector r_{0}~~\n");
            // print_vector(r_d, N);
        }

        // delta_old <- delta_new
        delta_old = delta_new;
        // //✅
        // printf("delta_old %f", delta_old);

        // delta_new <- r'_{i+1} * r_{i+1}
        checkCudaErrors(cublasSdot(cublasHandle, N, r_d, strd_x, r_d, strd_y, &delta_new));
        //✅
        // printf("delta_new %f", delta_new);
        cudaDeviceSynchronize();


        // bta <- delta_new / delta_old
        bta = delta_new / delta_old;
        //✅
        // printf("bta %f", bta);

        //ßd <- bta * d_{i}
        checkCudaErrors(cublasSscal(cublasHandle, N, &bta, dirc_d, strd_x));
        //✅
        // printf("\n\n~~ ß // bta * dirc_d_{i}~~\n");
        // print_vector(dirc_d, N);

        // d_{i+1} <- r_{i+1} + ßd_{i}
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, r_d, strd_x, dirc_d, strd_y));
        //✅
        // printf("\n\n~~vector dirc_d{i+1}~~\n");
        // print_vector(dirc_d, N);
        cudaDeviceSynchronize();


        cntr++;
    } // end of while

    if(cntr < MAX_ITR){
        printf("Converged at iteration %d", cntr);
    }else{
        printf("😫😫😫The iteration did not converged😫😫😫");
    }

    printf("\n\n~~vector x_sol~~\n");
    print_vector(x_d, N);
    
    float* x_h = (float*)malloc(sizeof(float) * N);
    CHECK(cudaMemcpy(x_h, x_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    //Check error as error = b - A * x_sol
    validate(mtxA_h, x_h, rhs, N);



    //(6) Free the GPU memory after use
    free(x_h);
    free(x);
    free(rhs);
    cudaFree(mtxA_d);
    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(dirc_d);
    cudaFree(Ax_d);
    cudaFree(q_d);

    return 0;
} // end of main