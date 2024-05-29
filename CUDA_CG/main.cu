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


//Hardcorded 3 by 3 matrix
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
    float *mtxA_d = nullptr;
    float *x_d = nullptr;  // Solution vector x
    float *r_d = nullptr; // Residual
    float *dirc_d = nullptr; // Direction
    float *Ax_d = nullptr; // Vector Ax
    float *q_d = nullptr; // Vector Ad
    float dot = 0.0f; // temporary val for d^{T} *q to get aplha
    
    //Using for cublas functin argument
    float alpha = 1.0; 
    float alphamns1 = -1.0;// negative alpha
    float beta = 0.0;

    float initial_delta = 0.0;
    float delta_new = 0.0;
    float delta_old = 0.0;
    float relative_residual = 0.0;
    const float EPS = 1e-5f;
    const int MAX_ITR = 4;

    // In CG iteration alpha and beta
    float alph = 0.0f;
    float ngtAlph = 0.0f;
    float bta = 0.0f;
    
    //Stride for x and y using contiguous vectors
    //Using for calling cublasSgemv
    int strd_x = 1;
    int strd_y = 1;
     
    // //Print 3 by 3 Matrix
    // printf("\n\n~~ 3 x 3 SPD matrix~~\n");
    // for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
    //     for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
    //         printf("%f ", mtxA_h[rw_wkr * N + clm_wkr]);
    //     }
    //     printf("\n");
    // }

    //Generating Random Dense SPD Matrix
    // float* mtxA_h = generateSPD_DenseMatrix(N);

    //Generating Random Tridiagonal SPD Matrix
    // float* mtxA_h = generate_TriDiagMatrix(N);
    // for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
    //     for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
    //         printf("%f ", mtxA_h[rw_wkr*N + clm_wkr]);
    //     }
    //     printf("\n");
    // }



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
    CHECK(cudaMalloc((void**)&dirc_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&Ax_d, sizeof(float) * N));
    CHECK(cudaMalloc((void**)&q_d, sizeof(float) * N));


    //(2) Copy value from host to device 
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, sizeof(float) * (N * N), cudaMemcpyHostToDevice)); 
    //✅
    // Print the matrix from device to host (Check for Debugging)
    print_mtx(mtxA_d, N, (N * N));


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
    // printf("\n\nr_{0} AKA given vector b = \n");
    // print_vector(r_d, N);


    //(3) Handle to the CUBLAS context
    //The cublasHandle variable will be used to store the handle to the cuBLAS library context. 
    cublasHandle_t cublasHandle = 0;
    cublasCreate(&cublasHandle);



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
    // printf("\n\nAx_{0} = \n");
    // print_vector(Ax_d, N);

    //2. Find residual r_{0} = b - Ax{0}
    //This function performs the operation y=αx+y.
    // Update the residual vector r by calculating r_{0} = b - Ax_{0}
    // Given vector b is d_r only used 1 time. We will updateing d_r as a new residual.
    // which is critical for determining the direction and magnitude of the initial search step in the CG algorithm.
    checkCudaErrors(cublasSaxpy(cublasHandle, N, &alphamns1, Ax_d, strd_x, r_d, strd_y));
    // //✅
    // printf("\n\nr_{0} = \n");
    // print_vector(r_d, N);

    //3. Set d <- r;
    CHECK(cudaMemcpy(dirc_d, r_d, N * sizeof(float), cudaMemcpyDeviceToDevice));
    // //✅
    // printf("\n\nd_{0}= \n");
    // print_vector(dirc_d, N);

    //4,  delta_{new} <- r^{T} * r
    // Compute the squared norm of the initial residual vector r (stored in r1).
    checkCudaErrors(cublasSdot(cublasHandle, N, r_d, strd_x, r_d, strd_y, &delta_new));
    //Save it for the relative residual calculation.
    initial_delta = delta_new;
    // //✅
    // printf("\n\ndelta_new{0} = \n %f\n ", delta_new);
    
    int cntr = 1; // counter

    bool debug = true;
    while(delta_new > EPS * EPS && cntr <= MAX_ITR){
        if(debug){
            printf("\n\n💫💫💫= = = Iteraion %d= = = 💫💫💫\n", cntr);
        }
        
        //q <- Ad
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, dirc_d, strd_x, &beta, q_d, strd_y));
        // //✅
        if(debug){
            printf("\nq = \n");
            print_vector(q_d, N);
        }
        

        //dot <- d^{T} * q
        checkCudaErrors(cublasSdot(cublasHandle, N, dirc_d, strd_x, q_d, strd_y, &dot));
        // //✅
        // if(debug){
        //     printf("\n\n~~dot AKA (d^{T*q)~~\n %f\n", dot);
        // }
        

        //alpha(a) <- delta_{new} / dot // dot <- d^{T} * q 
        alph = delta_new / dot;
        // //✅
        if(debug){
            printf("\nalpha = %f\n", alph);
        }
        

        //x_{i+1} <- x_{i} + alpha * d_{i}
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alph, dirc_d, strd_x, x_d, strd_y));
        // //✅

        if(debug){
            printf("\nx_sol = \n");
            print_vector(x_d, N);
        }


        if(cntr % 50 == 0){
            //r <- b -Ax Recompute
            // printf("\n\n= = = Iteration %d = = = = \n", cntr);

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
            if(debug && cntr == 3){
                printf("\n\n~~Before _{i+1} <- r_{i} -alpha*q~~\n");
                printf("\n\nr = \n");
                print_vector(r_d, N);
                printf("\n\nalpha = %f\n", alph);
                printf("\n\nngtAlpha = %f\n", ngtAlph);
                printf("\n\nq = \n");
                print_vector(q_d, N);
            }

            checkCudaErrors(cublasSaxpy(cublasHandle, N, &ngtAlph, q_d, 1, r_d, 1));
            //✅
            if(debug && cntr == 3){
                printf("\n\n~~After _{i+1} <- r_{i} -alpha*q~~\n");
                printf("\n\nr = \n");
                print_vector(r_d, N);
                printf("\n\nalpha = %f\n", alph);
                printf("\n\nq = \n");
                print_vector(q_d, N);
            }
            
        }

        // delta_old <- delta_new
        delta_old = delta_new;
        //✅
        if(debug){
            printf("\n\ndelta_old = %f\n", delta_old);
        }
        

        // delta_new <- r'_{i+1} * r_{i+1}
        checkCudaErrors(cublasSdot(cublasHandle, N, r_d, strd_x, r_d, strd_y, &delta_new));
        //✅
        if(debug){
            printf("\n\ndelta_new = %f\n", delta_new);
        }

        // bta <- delta_new / delta_old
        bta = delta_new / delta_old;
        //✅
        if(debug){
            printf("\nbta = %f\n", bta);
        }
        

        //ßd <- bta * d_{i}
        checkCudaErrors(cublasSscal(cublasHandle, N, &bta, dirc_d, strd_x));
        //✅
        // printf("\n\n~~ ß AKA bta * dirc_d_{i} = \n");
        // print_vector(dirc_d, N);

        // d_{i+1} <- r_{i+1} + ßd_{i}
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, r_d, 1, dirc_d, 1));
        //✅
        if(debug){
            printf("\nd = \n");
            print_vector(dirc_d, N);
        }

        relative_residual = sqrt(delta_new)/sqrt(initial_delta);
        printf("\n\nRelative residual = %f\n", relative_residual);
       
        cudaDeviceSynchronize();



        cntr++;
    } // end of while

    // if(cntr < MAX_ITR){
    //     printf("\n\n✅✅✅Converged at iteration %d✅✅✅\n", cntr-1);
    //     printf("\nRelative Error: delta_new = %f\n", delta_new);
    // }else{
    //     printf("\n\n😫😫😫The iteration did not converged😫😫😫\n");
    //     printf("\nRelative Error: delta_new = %f\n", delta_new);
    // }

    if(debug){
        printf("\n\nIteration %d", cntr - 1);
        printf("\nRelative residual = %f\n", relative_residual);
        printf("\n\n~~vector x_sol~~\n");
        print_vector(x_d, N);
    }
    
    
    float* x_h = (float*)malloc(sizeof(float) * N);
    CHECK(cudaMemcpy(x_h, x_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    //Check error as error = b - A * x_sol
    validate(mtxA_h, x_h, rhs, N);



    //(6) Free the GPU memory after use
    cudaFree(mtxA_d);
    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(dirc_d);
    cudaFree(Ax_d);
    cudaFree(q_d);
    cublasDestroy(cublasHandle);
    free(x_h);
    free(x);
    free(rhs);
    
    return 0;
} // end of main

