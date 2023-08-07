#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SCUDA_HOSTDEVICE __host__ __device__
#    define SCUDA_INLINE __forceinline__
#else
#    define SCUDA_HOSTDEVICE
#    define SCUDA_INLINE inline
#endif



SCUDA_INLINE SCUDA_HOSTDEVICE double dot(const double2& a, const double2& b)
{
    return a.x * b.x + a.y * b.y;
}

SCUDA_INLINE SCUDA_HOSTDEVICE double dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z ;
}


SCUDA_INLINE SCUDA_HOSTDEVICE double length(const double2& v)
{
    return sqrtf(dot(v, v));
}

SCUDA_INLINE SCUDA_HOSTDEVICE double length(const double3& v)
{
    return sqrtf(dot(v, v));
}






#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

inline std::ostream& operator<<(std::ostream& os, const double4& v)
{
    int w = 8 ; 
    os 
       << "(" 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.x 
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.y
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.z 
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.w 
       << ") "  
       ;
    return os; 
}

inline std::ostream& operator<<(std::ostream& os, const double3& v)
{
    int w = 8 ; 
    os 
       << "(" 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.x 
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.y
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.z 
       << ") "  
       ;
    return os; 
}

inline std::ostream& operator<<(std::ostream& os, const double2& v)
{
    int w = 8 ; 
    os 
       << "(" 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.x 
       << "," 
       << std::setw(w) << std::fixed << std::setprecision(4) << v.y
       << ") "  
       ;
    return os; 
}






#endif
