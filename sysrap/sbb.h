#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API sbb
{
    static constexpr const char* NAME = "sbb" ; 
    static constexpr const int N = 6 ; 
    double x0, y0, z0, x1, y1, z1 ;  
    std::string desc() const ; 
}; 

inline std::string sbb::desc() const
{
    const double* d = &x0 ; 
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << d[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}


