#include <cmath>
#include <cstdlib>
#include <iostream>

#include "kernels.h"

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Check that all values of array (which contains `length` float elements) are
// close to `expectedValue`
bool checkHostArray(float *array, float expectedValue, size_t length)
{
    float maxError = 0.0f;
    for (int i = 0; i < length; i++)
        maxError = fmax(maxError, fabs(array[i] - expectedValue));
    std::cout << "Max error: " << maxError << std::endl;
    return (maxError < 0.0001f);
}

int main(void)
{
    int N = 1 << 20; //< Number of elements in arrays (1M, you may want to lower
                     //           this to begin)
    float *d_x; //< Pointer to the 1D buffer we will manipulate

    // Initialize the x and y arrays on the device
    // use as many threads as possible,
    // and assume we will process each element of d_x
    // with a different thread
    float *d_y;

    //@@ Choose some values here, stick to 1D
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1);

    printf("DIM - x: %d, y:%d\n", threadsPerBlock, blocksPerGrid);

    // Array allocation on device
    //@@ Use cudaMalloc to perform the allocation.
    //@@ cudaMalloc(???); // FIXME
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize the x and y arrays on the device
    const float firstValue = 1.f;
    //@@ Call the fill1D kernel to fill d_x with `firstValue`, see kernels.h for
    //           the API
    //@@ fill1D<<<???, ???>>>(???);  // FIXME
    // Wait for GPU to finish and check for errors

    fill1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue, N);

    cudaDeviceSynchronize();
    cudaCheckError();

    // Check for errors on device
    float expectedValue = firstValue;
    //@@ Call the check1D kernel to control device memory content, see kernels.h
    //           for API
    //@@ check1D<<<?? ,???>>>(???);  // FIXME
    // Wait for GPU to finish and check for errors
    //@@ call CUDA device synchronisation function
    //@@ ???

    check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, expectedValue, N);

    cudaDeviceSynchronize();
    cudaCheckError();

    // Copy back the buffer to the host for inspection:
    //@@ Allocate a buffer on the host
    //@@ float *h_x = (float*) std::malloc(???);  //FIXME
    //@@ Copy the buffer content from device to host
    //@@ use cudaMemcpy
    //@@ cudaMemcpy(???);  // FIXME
    float *h_x = (float*) std::malloc(N * sizeof(float));
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckError();

    // Check for errors (all values should be close to `firstValue`)
    std::cout << "First control..." << std::endl;
    bool noerror = checkHostArray(h_x, firstValue, N);

    // Now increment the array values by some other value
    const float otherValue = 10.f;
    //   @@ Call the inc1D kernel to add `otherValue` to all values of our buffer,
    //        see kernels.h for API
    //@@ inc1D<<<???, ???>>>(???);
    // Wait for GPU to finish
    //@@ call CUDA device synchronisation function
    //@@ ???

    inc1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, otherValue, N);
    
    cudaDeviceSynchronize();
    cudaCheckError();

    // Check for errors on device
    expectedValue = firstValue + otherValue;
    //   @@ Call the check1D kernel to control device memory content, see kernels.h
    //        for API
    //@@ check1D<<<?? ,???>>>(???);  // FIXME
    // Wait for GPU to finish and check for errors
    //@@ call CUDA device synchronisation function
    //@@ ???

    check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, expectedValue, N);

    cudaDeviceSynchronize();
    cudaCheckError();

    // Copy back the buffer to the host for inspection:
    //@@ Copy the buffer content from device to host (reuse previous buffer)
    //@@ cudaMemcpy(????);  // FIXME

    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Check for errors (all values should be close to `firstValue+otherValue`)
    std::cout << "Second control..." << std::endl;
    noerror &= checkHostArray(h_x, firstValue + otherValue, N);

    // Free memory
    //@@ free d_h using CUDA primitives
    //@@ cuda???
    cudaFree(d_x);
    cudaFree(d_y);
    cudaCheckError();
    std::free(h_x);

    if (noerror)
    {
        printf("Test completed successfully.\n");
        return 0;
    }
    else
    {
        printf("WARNING there were some errors.\n");
        return 1;
    }
}
