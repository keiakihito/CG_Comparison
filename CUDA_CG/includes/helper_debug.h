#ifndef HELPER_DEBUG_H
#define HELPER_DEBUG_H

#include <iostream>
#include <cuda_runtime.h>

void print_vector(const float *d_val, int size) {
    // Allocate memory on the host
    float *check_r = (float *)malloc(sizeof(float) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }
    // Print the values to check them
    for (int i = 0; i < size; i++) {
        if( i < 3){
            printf("%f \n", check_r[i]);
        }
    }
    // printf("\n...\n");
    // printf("%f ", check_r[size-3]);
    // printf("%f ", check_r[size-2]);
    // printf("%f ", check_r[size-1]);
    

    // Free allocated memory
    free(check_r);
} // print_vector

void print_vector(const int *d_val, int size) {
    // Allocate memory on the host
    float *check_r = (float *)malloc(sizeof(float) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }
    // Print the values to check them
    for (int i = 0; i < size; i++) {
        if( i < 3){
            printf("%d \n", check_r[i]);
        }
    }
    // printf("\n...\n");
    // printf("%f ", check_r[size-3]);
    // printf("%f ", check_r[size-2]);
    // printf("%f ", check_r[size-1]);
    

    // Free allocated memory
    free(check_r);
}// end of print_vector

//N is row and cloumn size 
void print_mtx(const float *d_val, int N, int size){
    //Allocate memory oh the host
    float *check_r = (float *)malloc(sizeof(float) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }

    // Print the values to check them
    for (int i = 0; i < size; i++) {
        if( i % N == 0 && i != 0){
            printf("\n");
        }
        printf("%f ", check_r[i]);
    }
    printf("\n\n");

    // Free allocated memory
    free(check_r);
} // end of print_mtx

#endif // HELPER_DEBUG_H
