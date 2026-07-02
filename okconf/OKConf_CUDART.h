#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <iomanip>

struct OKConf_CUDART
{
    static int         CUDADriverInteger();
    static std::string Desc();
    static int         Check();
};

inline int OKConf_CUDART::CUDADriverInteger()
{
    int driver_version = 0;
    cudaError_t err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        std::cerr << "OKConf::CUDADriverInteger - Warning: Could not resolve runtime CUDA driver: "
                  << cudaGetErrorString(err) << "\n";
    }
    return driver_version ;
}

inline std::string OKConf_CUDART::Desc()
{
    std::stringstream ss;
    ss << std::setw(50) << "OKConf_CUDART::CUDADriverInteger() "       << CUDADriverInteger() << std::endl ;
    std::string str = ss.str();
    return str ;
}


inline int OKConf_CUDART::Check()
{
   int rc = 0 ;
   if(CUDADriverInteger() == 0)
   {
       rc += 1 ;
   }
   return rc ;
}


