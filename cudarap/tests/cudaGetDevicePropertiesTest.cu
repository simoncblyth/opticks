#include <cstdio>
#include <cstring>
 
// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %zu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
 
int main(int argc, char** argv)
{
    // Number of CUDA devices
    int devCount;
    printf( " argv[0] %s cudaGetDeviceCount START \n", argv[0] );
    cudaGetDeviceCount(&devCount);
    printf( " argv[0] %s cudaGetDeviceCount DONE \n", argv[0] );

    bool quiet = argc > 1 && strlen(argv[1]) > 0 && argv[1][0] == 'q' ; 

    int target = argc > 1 && strlen(argv[1]) > 1 ? atoi(argv[1]+1) : -1 ; 


    if(!quiet)
    {
       printf("CUDA Device Query...target %d \n", target);
       printf("There are %d CUDA devices.\n", devCount);
       
    } 

    int compute_capability = 0 ; 

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        if(!quiet) printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        if(!quiet) printDevProp(devProp);

        compute_capability = devProp.major*10 + devProp.minor ;  
    }


    printf("%d\n", compute_capability);

    cudaDeviceSynchronize(); 
    return 0;
}
