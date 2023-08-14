#pragma once
/**
spa.h
======

last two param are zmin/zmax for multiple types::

   CSG_ZSPHERE
   CSG_CYLINDER
   CSG_CONE

**/
#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API spa
{
    static constexpr const char* NAME = "spa" ; 
    static constexpr const int N = 6 ; 
    double x0, y0, z0, w0, x1, y1 ;  

    const double* data() const { return &x0 ; }  
    double zmin() const { return x1 ; }
    double zmax() const { return y1 ; }
    void decrease_zmin(double dz){ assert( dz >= 0.) ; x1 -= dz ; }
    void increase_zmax(double dz){ assert( dz >= 0.) ; y1 += dz ; }

    std::string desc() const ; 
}; 

inline std::string spa::desc() const
{
    const double* v = data() ; 
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < N ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << v[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}


