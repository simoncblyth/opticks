#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define PARM_METHOD __device__ __host__ __forceinline__
   #define PARM_FUNCTION  __device__ __host__ __forceinline__
#else
   #define PARM_METHOD inline
   #define PARM_FUNCTION inline
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
   #include <sstream>
#endif

#include "squad.h"

struct sparm
{
    quad q0, q1 ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    PARM_METHOD void zero();
    PARM_METHOD void set6(const double* pa);
    PARM_METHOD std::string desc() const ;
#endif
};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

PARM_METHOD void sparm::zero()
{
    q0.f.x = 0.f ; q0.f.y = 0.f ; q0.f.z = 0.f ; q0.f.w = 0.f ;
    q1.f.x = 0.f ; q1.f.y = 0.f ; q1.f.z = 0.f ; q1.f.w = 0.f ;
}

PARM_METHOD void sparm::set6(const double* pa)
{
    q0.f.x = *(pa + 0);
    q0.f.y = *(pa + 1);
    q0.f.z = *(pa + 2);
    q0.f.w = *(pa + 3);

    q1.f.x = *(pa + 4);
    q1.f.y = *(pa + 5);
    q1.f.z = 0.f ;
    q1.f.w = 0.f ;
}

PARM_METHOD std::string sparm::desc() const
{
    std::stringstream ss;
    ss << "[sparm" ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q0.f.x ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q0.f.y ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q0.f.z ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q0.f.w ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q1.f.x ;
    ss << " " << std::fixed << std::setw(7) << std::setprecision(3) << q1.f.y ;
    ss << "] " ;
    std::string str = ss.str();
    return str ;
}


#endif



