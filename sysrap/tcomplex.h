#pragma once
/**
tcomplex.h
=============

Based on Yuxiangs implementation following ROOT TComplex.h
https://root.cern.ch/doc/master/TComplex_8h_source.html

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define TCOMPLEX_METHOD __host__ __device__
#else
   #define TCOMPLEX_METHOD 
#endif 


#include <cuComplex.h>

struct tcomplex
{
    static TCOMPLEX_METHOD cuFloatComplex cuSqrtf( const cuFloatComplex& a) ; 
    static TCOMPLEX_METHOD cuFloatComplex make_cuFloatComplex_polar(float radius , float theta) ; 
    static TCOMPLEX_METHOD float          cuRhof(  const cuFloatComplex& a) ; 
    static TCOMPLEX_METHOD float          cuThetaf(const cuFloatComplex& a) ; 
};

inline TCOMPLEX_METHOD cuFloatComplex tcomplex::make_cuFloatComplex_polar(float radius , float theta)
{
    return make_cuFloatComplex(fabsf(radius)*cosf(theta), fabsf(radius)*sinf(theta));
}
inline TCOMPLEX_METHOD float tcomplex::cuRhof(const cuFloatComplex& a)
{
    return sqrtf( a.x*a.x + a.y*a.y ) ;
}
inline TCOMPLEX_METHOD float tcomplex::cuThetaf(const cuFloatComplex& a)
{
    return (a.x||a.y) ? atan2f(a.y,a.x):0.f ;
}
inline TCOMPLEX_METHOD cuFloatComplex tcomplex::cuSqrtf(const cuFloatComplex& a)
{
    return make_cuFloatComplex_polar( sqrtf(cuRhof(a)) , 0.5f*cuThetaf(a) ); 
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <iostream>
#include <iomanip>

/*
//
// This symbol is already provided by scuda.h and hence gives "error redefinition" 
// as from /usr/local/cuda/include/cuComplex.h cuFloatComplex is typedef to float2
//

inline std::ostream& operator<<(std::ostream& os, const cuFloatComplex& a)
{
    int w = 6 ; 
    os 
       << "(" 
       << std::setw(w) << std::fixed << std::setprecision(3) << a.x 
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(3) << a.y 
       << ") "  
       ;
    return os; 
}
*/

#endif


