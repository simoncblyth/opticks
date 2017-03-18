#pragma once

#include "NQuad.hpp"
#include "NNode.hpp"

struct nplane ; 
struct ndisc ; 
struct npart ;

#include "NPY_API_EXPORT.hh"

struct NPY_API nsphere : nnode {

    // NO CTOR

    float x();
    float y();
    float z();
    float radius();
    float costheta(float z);

    double operator()(double px, double py, double pz) ;
 
    npart part();
    static ndisc intersect(nsphere& a, nsphere& b);

    // result of intersect allows partitioning 
    npart zrhs(const ndisc& dsc); // +z to the right  
    npart zlhs(const ndisc& dsc);  

    void dump(const char* msg);

    nvec4 param ; 
};


inline NPY_API nsphere make_nsphere(float x, float y, float z, float w)
{
    nsphere s ; s.param.x = x ; s.param.y = y ; s.param.z = z ; s.param.w = w ; return s ;
}

inline NPY_API nsphere make_nsphere(const nvec4& p)
{
    nsphere s ; s.param.x = p.x ; s.param.y = p.y ; s.param.z = p.z ; s.param.w = p.w ; return s ;
}


