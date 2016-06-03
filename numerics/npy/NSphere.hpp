#pragma once

#include "NQuad.hpp"

struct nplane ; 
struct ndisc ; 
struct npart ;

struct nsphere {

    // NO CTOR

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


inline nsphere make_nsphere(float x, float y, float z, float w)
{
    nsphere s ; s.param.x = x ; s.param.y = y ; s.param.z = z ; s.param.w = w ; return s ;
}

inline nsphere make_nsphere(const nvec4& p)
{
    nsphere s ; s.param.x = p.x ; s.param.y = p.y ; s.param.z = p.z ; s.param.w = p.w ; return s ;
}



inline float nsphere::radius(){ return param.w ; }
inline float nsphere::x(){      return param.x ; }
inline float nsphere::y(){      return param.y ; }
inline float nsphere::z(){      return param.z ; }

inline float nsphere::costheta(float z)
{
   return (z - param.z)/param.w ;  
}
