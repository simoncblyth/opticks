/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

//  cudarap/tests/cudaGetDevicePropertiesTest.cu
//

#include "UseCUDA.h"

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

 
int UseCUDA_query_device(int argc, char** argv)
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




