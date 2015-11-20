#pragma once

#include "NQuad.hpp"

struct nplane ; 
struct ndisc ; 
struct npart ;

struct nsphere {
    nsphere(float x, float y, float z, float w);
    nsphere(const nvec4& param_);

    float x();
    float y();
    float z();
    float radius();
    float costheta(float z);

    npart part();
    static ndisc intersect(nsphere& a, nsphere& b);

    // result of intersect allows partitioning 
    npart zrhs(const ndisc& dsc); // +z to the right  
    npart zlhs(const ndisc& dsc);  

    void dump(const char* msg);

    nvec4 param ; 
};


inline nsphere::nsphere(float x, float y, float z, float w)
{
    param.x = x  ;
    param.y = y  ;
    param.z = z  ;
    param.w = w  ;
}

inline nsphere::nsphere(const nvec4& param_)
{
    param = param_ ;
}

inline float nsphere::radius(){ return param.w ; }
inline float nsphere::x(){      return param.x ; }
inline float nsphere::y(){      return param.y ; }
inline float nsphere::z(){      return param.z ; }

inline float nsphere::costheta(float z)
{
   return (z - param.z)/param.w ;  
}
