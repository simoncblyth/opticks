#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API sbb
{
    static constexpr const char* NAME = "sbb" ; 
    static constexpr const int N = 6 ; 
    static bool IsZero( const double* v ); 

    double x0, y0, z0, x1, y1, z1 ;  

    double* data() { return &x0 ; }  
    const double* cdata() const { return &x0 ; }  
    double zmin() const { return z0 ; }
    double zmax() const { return z1 ; }
 
    void increase_zmax(double dz) { assert( dz >= 0. ) ; z1 += dz ; }
    void decrease_zmin(double dz) { assert( dz >= 0. ) ; z0 -= dz ; }

    std::string desc() const ; 
    static std::string Desc(const double* d); 
}; 

inline bool sbb::IsZero( const double* v )
{
    int count = 0 ; 
    for(int i=0 ; i < N ; i++) if(v[i] == 0.) count += 1 ; 
    return count == N ; 
}

inline std::string sbb::desc() const
{
    const double* v = cdata() ; 
    return Desc(v) ; 
}

inline std::string sbb::Desc(const double* v) 
{
    int wid = 8 ; 
    int pre = 3 ; 
    std::stringstream ss ;
    ss << "(" ;  
    for(int i=0 ; i < 6 ; i++) ss << std::setw(wid) << std::fixed << std::setprecision(pre) << v[i] << " " ; 
    ss << ")" ;  
    std::string str = ss.str(); 
    return str ; 
}


