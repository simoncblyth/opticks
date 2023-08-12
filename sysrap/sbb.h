#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "SYSRAP_API_EXPORT.hh"


template<typename T>
struct SYSRAP_API sbb
{
    static constexpr const char* NAME = "sbb" ; 
    static constexpr const int N = 6 ; 
    T x0, y0, z0, x1, y1, z1 ;  

    T zmin() const { return z0 ; }
    T zmax() const { return z1 ; }
 
    void increase_zmax(T dz) { assert( dz >= T(0.)) ; z1 += dz ; }
    void decrease_zmin(T dz) { assert( dz >= T(0.)) ; z0 -= dz ; }

    std::string desc() const ; 
}; 

template<typename T>
inline std::string sbb<T>::desc() const
{
    const T* v = &x0 ; 
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << v[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}


