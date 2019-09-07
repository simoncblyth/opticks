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

#include <cstdio>
#include <vector>

#include "cuda.h"
#include "driver_types.h"   // for cudaError_t
#include "helper_cuda.h"    // for _cudaGetErrorEnum


int main()
{
    printf(" CUDA_VERSION  %d \n", CUDA_VERSION ) ; 

    std::vector<cudaError_t> errs ; 

    errs.push_back(cudaSuccess); 
    errs.push_back(cudaErrorLaunchFailure); 
    errs.push_back(cudaErrorLaunchTimeout); 
 
    for(unsigned i=0 ; i < errs.size() ; i++)
    {
        cudaError_t err = errs[i] ; 
        const char* err_ = _cudaGetErrorEnum(err) ; 

        printf(" %4d %s \n", err, err_ ? err_ : "?" );
   }

    return 0 ; 
}
