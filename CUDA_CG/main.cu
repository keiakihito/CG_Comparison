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
#define N 32000 // 2^15 < 46340 < 2^16


// //Hardcorded 3 by 3 matrix
// float mtxA_h[N*N] = {
//     1.5004, 1.3293, 0.8439,
//     1.3293, 1.2436, 0.6936,
//     0.8439, 0.6936, 1.2935
// };

// // Hardcoded 18 by 18 matrix
// float mtxA_h[N * N] = {
//     9.3918, 6.6007, 6.3940, 6.2324, 6.0555, 4.9198, 5.9791, 4.3268, 5.6480, 5.2860, 7.4877, 4.5666, 4.9256, 6.4601, 6.5433, 6.0055, 6.4163, 4.9032,
//     6.6007, 6.9943, 5.2786, 5.2829, 5.1832, 3.1949, 5.2337, 4.1912, 4.8611, 4.9841, 5.9790, 4.1163, 4.3596, 4.9439, 5.3257, 5.0629, 5.1506, 4.6618,
//     6.3940, 5.2786, 6.6018, 4.6547, 5.5515, 4.2225, 4.7058, 3.7232, 4.6784, 4.0535, 5.7858, 3.8962, 4.4437, 4.9393, 5.1749, 4.5644, 5.7526, 5.1598,
//     6.2324, 5.2829, 4.6547, 6.4837, 5.1888, 4.0699, 5.3413, 3.9317, 4.7237, 4.8594, 6.1045, 4.0298, 3.5701, 4.8748, 5.6800, 4.0319, 5.3133, 4.5944,
//     6.0555, 5.1832, 5.5515, 5.1888, 6.4653, 4.3358, 4.9567, 3.8437, 4.2128, 4.5556, 5.7876, 4.0070, 4.4707, 4.5421, 5.7059, 4.2554, 5.9133, 5.2349,
//     4.9198, 3.1949, 4.2225, 4.0699, 4.3358, 4.7261, 3.3840, 2.2092, 3.7733, 3.3743, 4.9615, 2.9836, 3.2188, 3.8934, 4.3613, 3.3114, 4.5301, 3.8181,
//     5.9791, 5.2337, 4.7058, 5.3413, 4.9567, 3.3840, 7.0722, 4.0417, 4.5106, 4.7280, 5.7858, 4.1902, 3.9915, 4.9707, 5.7849, 5.0159, 4.7386, 4.3284,
//     4.3268, 4.1912, 3.7232, 3.9317, 3.8437, 2.2092, 4.0417, 3.5335, 3.1209, 3.0387, 3.8865, 2.7753, 2.7530, 3.7246, 3.9503, 3.3912, 3.9507, 3.2583,
//     5.6480, 4.8611, 4.6784, 4.7237, 4.2128, 3.7733, 4.5106, 3.1209, 6.4609, 4.1038, 6.0113, 4.2932, 4.1595, 5.5648, 4.8813, 4.1930, 4.1077, 4.0631,
//     5.2860, 4.9841, 4.0535, 4.8594, 4.5556, 3.3743, 4.7280, 3.0387, 4.1038, 5.0346, 5.2025, 3.6426, 3.6059, 3.7162, 4.6563, 3.8483, 3.9120, 3.6490,
//     7.4877, 5.9790, 5.7858, 6.1045, 5.7876, 4.9615, 5.7858, 3.8865, 6.0113, 5.2025, 7.9688, 5.4157, 4.6194, 6.4850, 6.7675, 5.3269, 6.1117, 5.3438,
//     4.5666, 4.1163, 3.8962, 4.0298, 4.0070, 2.9836, 4.1902, 2.7753, 4.2932, 3.6426, 5.4157, 4.9784, 3.4655, 5.0211, 4.5979, 3.8621, 3.6243, 3.8400,
//     4.9256, 4.3596, 4.4437, 3.5701, 4.4707, 3.2188, 3.9915, 2.7530, 4.1595, 3.6059, 4.6194, 3.4655, 4.5696, 4.1065, 3.8278, 3.7711, 3.8671, 3.8748,
//     6.4601, 4.9439, 4.9393, 4.8748, 4.5421, 3.8934, 4.9707, 3.7246, 5.5648, 3.7162, 6.4850, 5.0211, 4.1065, 6.8418, 5.6289, 5.1061, 4.9464, 4.1055,
//     6.5433, 5.3257, 5.1749, 5.6800, 5.7059, 4.3613, 5.7849, 3.9503, 4.8813, 4.6563, 6.7675, 4.5979, 3.8278, 5.6289, 7.0175, 4.3347, 5.5875, 4.9894,
//     6.0055, 5.0629, 4.5644, 4.0319, 4.2554, 3.3114, 5.0159, 3.3912, 4.1930, 3.8483, 5.3269, 3.8621, 3.7711, 5.1061, 4.3347, 6.0332, 4.6887, 3.1360,
//     6.4163, 5.1506, 5.7526, 5.3133, 5.9133, 4.5301, 4.7386, 3.9507, 4.1077, 3.9120, 6.1117, 3.6243, 3.8671, 4.9464, 5.5875, 4.6887, 6.8489, 5.0721,
//     4.9032, 4.6618, 5.1598, 4.5944, 5.2349, 3.8181, 4.3284, 3.2583, 4.0631, 3.6490, 5.3438, 3.8400, 3.8748, 4.1055, 4.9894, 3.1360, 5.0721, 5.6506
// };

// // Hardcoded 19 by 19 matrix
//     // Example hardcoded matrix and vectors
//     float mtxA_h[N * N] = {
//         9.3918, 6.6007, 6.3940, 6.2324, 6.0555, 4.9198, 5.9791, 4.3268, 5.6480, 5.2860, 7.4877, 4.5666, 4.9256, 6.4601, 6.5433, 6.0055, 6.4163, 4.9032, 5.5738,
//         6.6007, 6.9943, 5.2786, 5.2829, 5.1832, 3.1949, 5.2337, 4.1912, 4.8611, 4.9841, 5.9790, 4.1163, 4.3596, 4.9439, 5.3257, 5.0629, 5.1506, 4.6618, 5.3536,
//         6.3940, 5.2786, 6.6018, 4.6547, 5.5515, 4.2225, 4.7058, 3.7232, 4.6784, 4.0535, 5.7858, 3.8962, 4.4437, 4.9393, 5.1749, 4.5644, 5.7526, 5.1598, 4.5692,
//         6.2324, 5.2829, 4.6547, 6.4837, 5.1888, 4.0699, 5.3413, 3.9317, 4.7237, 4.8594, 6.1045, 4.0298, 3.5701, 4.8748, 5.6800, 4.0319, 5.3133, 4.5944, 4.7613,
//         6.0555, 5.1832, 5.5515, 5.1888, 6.4653, 4.3358, 4.9567, 3.8437, 4.2128, 4.5556, 5.7876, 4.0070, 4.4707, 4.5421, 5.7059, 4.2554, 5.9133, 5.2349, 5.0687,
//         4.9198, 3.1949, 4.2225, 4.0699, 4.3358, 4.7261, 3.3840, 2.2092, 3.7733, 3.3743, 4.9615, 2.9836, 3.2188, 3.8934, 4.3613, 3.3114, 4.5301, 3.8181, 3.6113,
//         5.9791, 5.2337, 4.7058, 5.3413, 4.9567, 3.3840, 7.0722, 4.0417, 4.5106, 4.7280, 5.7858, 4.1902, 3.9915, 4.9707, 5.7849, 5.0159, 4.7386, 4.3284, 5.2583,
//         4.3268, 4.1912, 3.7232, 3.9317, 3.8437, 2.2092, 4.0417, 3.5335, 3.1209, 3.0387, 3.8865, 2.7753, 2.7530, 3.7246, 3.9503, 3.3912, 3.9507, 3.2583, 3.8240,
//         5.6480, 4.8611, 4.6784, 4.7237, 4.2128, 3.7733, 4.5106, 3.1209, 6.4609, 4.1038, 6.0113, 4.2932, 4.1595, 5.5648, 4.8813, 4.1930, 4.1077, 4.0631, 4.3061,
//         5.2860, 4.9841, 4.0535, 4.8594, 4.5556, 3.3743, 4.7280, 3.0387, 4.1038, 5.0346, 5.2025, 3.6426, 3.6059, 3.7162, 4.6563, 3.8483, 3.9120, 3.6490, 4.8429,
//         7.4877, 5.9790, 5.7858, 6.1045, 5.7876, 4.9615, 5.7858, 3.8865, 6.0113, 5.2025, 7.9688, 5.4157, 4.6194, 6.4850, 6.7675, 5.3269, 6.1117, 5.3438, 5.7159,
//         4.5666, 4.1163, 3.8962, 4.0298, 4.0070, 2.9836, 4.1902, 2.7753, 4.2932, 3.6426, 5.4157, 4.9784, 3.4655, 5.0211, 4.5979, 3.8621, 3.6243, 3.8400, 4.4969,
//         4.9256, 4.3596, 4.4437, 3.5701, 4.4707, 3.2188, 3.9915, 2.7530, 4.1595, 3.6059, 4.6194, 3.4655, 4.5696, 4.1065, 3.8278, 3.7711, 3.8671, 3.8748, 4.1143,
//         6.4601, 4.9439, 4.9393, 4.8748, 4.5421, 3.8934, 4.9707, 3.7246, 5.5648, 3.7162, 6.4850, 5.0211, 4.1065, 6.8418, 5.6289, 5.1061, 4.9464, 4.1055, 5.0408,
//         6.5433, 5.3257, 5.1749, 5.6800, 5.7059, 4.3613, 5.7849, 3.9503, 4.8813, 4.6563, 6.7675, 4.5979, 3.8278, 5.6289, 7.0175, 4.3347, 5.5875, 4.9894, 5.4625,
//         6.0055, 5.0629, 4.5644, 4.0319, 4.2554, 3.3114, 5.0159, 3.3912, 4.1930, 3.8483, 5.3269, 3.8621, 3.7711, 5.1061, 4.3347, 6.0332, 4.6887, 3.1360, 5.1251,
//         6.4163, 5.1506, 5.7526, 5.3133, 5.9133, 4.5301, 4.7386, 3.9507, 4.1077, 3.9120, 6.1117, 3.6243, 3.8671, 4.9464, 5.5875, 4.6887, 6.8489, 5.0721, 4.6645,
//         4.9032, 4.6618, 5.1598, 4.5944, 5.2349, 3.8181, 4.3284, 3.2583, 4.0631, 3.6490, 5.3438, 3.8400, 3.8748, 4.1055, 4.9894, 3.1360, 5.0721, 5.6506, 3.6533,
//         5.5738, 5.3536, 4.5692, 4.7613, 5.0687, 3.6113, 5.2583, 3.8240, 4.3061, 4.8429, 5.7159, 4.4969, 4.1143, 5.0408, 5.4625, 5.1251, 4.6645, 3.6533, 6.4536
//     };






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
    float* mtxA_h = generate_TriDiagMatrix(N);
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
    // // Print the matrix from device to host (Check for Debugging)
    // print_mtx(mtxA_d, N, (N * N));


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
    // //✅
    // printf("\n\ndelta_new{0} = \n %f\n ", delta_new);
    
    int cntr = 1; // counter

    bool debug = true;
    while(delta_new > EPS * EPS && cntr < MAX_ITR){
        if(debug){
            printf("\n\n= = = Iteraion %d= = = \n", cntr);
        }
        
        //q <- Ad
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, dirc_d, strd_x, &beta, q_d, strd_y));
        // //✅
        if(debug){
            printf("\nq = \n");
            print_vector(q_d, N);
        }
        

        //dot <- d^{T} * q
        checkCudaErrors((cublasSdot(cublasHandle, N, dirc_d, strd_x, q_d, strd_y, &dot)));
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
        checkCudaErrors((cublasSaxpy(cublasHandle, N, &alph, dirc_d, strd_x, x_d, strd_y)));
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
            checkCudaErrors(cublasSaxpy(cublasHandle, N, &ngtAlph, q_d, strd_x, r_d, strd_y));
            //✅
            if(debug){
                printf("\n\nr = \n");
                print_vector(r_d, N);
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
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, r_d, strd_x, dirc_d, strd_y));
        //✅
        if(debug){
            printf("\nd = \n");
            print_vector(dirc_d, N);
        }
       
        cudaDeviceSynchronize();


        cntr++;
    } // end of while

    if(cntr < MAX_ITR){
        printf("\n\n✅✅✅Converged at iteration %d✅✅✅\n", cntr-1);
        printf("\nRelative Error: delta_new = %f\n", delta_new);
    }else{
        printf("\n\n😫😫😫The iteration did not converged😫😫😫\n");
        printf("\nRelative Error: delta_new = %f\n", delta_new);
    }

    if(debug){
        printf("\n\nIteration %d", cntr - 1);
        printf("\nRelative Error: delta_new = %f\n", delta_new);
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


/*
Sample Run
~~ 3 x 3 SPD matrix~~
1.500400 1.329300 0.843900 
1.329300 1.243600 0.693600 
0.843900 0.693600 1.293500 


= = = Iteraion 1= = = 

q = 
3.673600 
3.266500 
2.831000 

alpha = 0.307028

x_sol = 
0.307028 
0.307028 
0.307028 

r = 
-0.127898 
-0.002907 
0.130804 

delta_old = 3.000000

delta_new = 0.033476

bta = 0.011159

d = 
-0.116739 
0.008252 
0.141963 


= = = Iteraion 2= = = 

q = 
-0.044384 
-0.046454 
0.090836 

alpha = 1.892012

x_sol = 
0.086156 
0.322641 
0.575623 

r = 
-0.043923 
0.084984 
-0.041059 

delta_old = 0.033476

delta_new = 0.010837

bta = 0.323739

d = 
-0.081716 
0.087656 
0.004900 


= = = Iteraion 3= = = 

q = 
-0.001952 
0.003781 
-0.001825 

alpha = 22.483810

x_sol = 
-1.751141 
2.293478 
0.685784 

r = 
-0.000041 
-0.000037 
-0.000030 

delta_old = 0.010837

delta_new = 0.000000

bta = 0.000000

d = 
-0.000041 
-0.000037 
-0.000030 


= = = Iteraion 4= = = 

q = 
-0.000137 
-0.000122 
-0.000099 

alpha = 0.302960

x_sol = 
-1.751154 
2.293466 
0.685775 

r = 
0.000000 
-0.000000 
0.000000 

delta_old = 0.000000

delta_new = 0.000000

bta = 0.000000

d = 
0.000000 
-0.000000 
0.000000 
Converged at iteration 5

~~vector x_sol~~
-1.751154 
2.293466 
0.685775 


Test Summary: Error amount = 0.000000
[kkatsumi@gpub080 CUDA_CG]$ nvcc main.cu -o main  -lcublas -lcusparse
[kkatsumi@gpub080 CUDA_CG]$ ./main


~~ 3 x 3 SPD matrix~~
1.500400 1.329300 0.843900 
1.329300 1.243600 0.693600 
0.843900 0.693600 1.293500 


= = = Iteraion 1= = = 

q = 
3.673600 
3.266500 
2.831000 

alpha = 0.307028

x_sol = 
0.307028 
0.307028 
0.307028 

r = 
-0.127898 
-0.002907 
0.130804 

delta_old = 3.000000

delta_new = 0.033476

bta = 0.011159

d = 
-0.116739 
0.008252 
0.141963 


= = = Iteraion 2= = = 

q = 
-0.044384 
-0.046454 
0.090836 

alpha = 1.892012

x_sol = 
0.086156 
0.322641 
0.575623 

r = 
-0.043923 
0.084984 
-0.041059 

delta_old = 0.033476

delta_new = 0.010837

bta = 0.323739

d = 
-0.081716 
0.087656 
0.004900 


= = = Iteraion 3= = = 

q = 
-0.001952 
0.003781 
-0.001825 

alpha = 22.483810

x_sol = 
-1.751141 
2.293478 
0.685784 

r = 
-0.000041 
-0.000037 
-0.000030 

delta_old = 0.010837

delta_new = 0.000000

bta = 0.000000

d = 
-0.000041 
-0.000037 
-0.000030 


= = = Iteraion 4= = = 

q = 
-0.000137 
-0.000122 
-0.000099 

alpha = 0.302960

x_sol = 
-1.751154 
2.293466 
0.685775 

r = 
0.000000 
-0.000000 
0.000000 

delta_old = 0.000000

delta_new = 0.000000

bta = 0.000000

d = 
0.000000 
-0.000000 
0.000000 
Converged at iteration 5

~~vector x_sol~~
-1.751154 
2.293466 
0.685775 


*/