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
