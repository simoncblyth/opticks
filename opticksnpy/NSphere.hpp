#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct nplane ; 
struct ndisc ; 
struct npart ;
struct nbbox ; 


#include "NPY_API_EXPORT.hh"

struct NPY_API nsphere : nnode {

    // NO CTOR

    float x();
    float y();
    float z();
    float radius();
    float costheta(float z);

    double operator()(double px, double py, double pz) ;
    nbbox bbox();

    npart part();
    static ndisc intersect(nsphere& a, nsphere& b);

    // result of intersect allows partitioning 
    npart zrhs(const ndisc& dsc); // +z to the right  
    npart zlhs(const ndisc& dsc);  

    glm::vec3 gcenter() ;
    void pdump(const char* msg="nsphere::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    float     radius_ ; 

};


inline NPY_API void init_nsphere(nsphere& s, const nvec4& param)
{
    s.param.x = param.x ; 
    s.param.y = param.y ; 
    s.param.z = param.z ; 
    s.param.w = param.w ; 

    s.center.x = param.x ; 
    s.center.y = param.y ; 
    s.center.z = param.z ;
    s.radius_  = param.w ;  
}

inline NPY_API nsphere make_nsphere(const nvec4& param)
{
    nsphere n ; 
    nnode::Init(n,CSG_SPHERE) ; 
    init_nsphere(n, param);
    return n ; 
}
inline NPY_API nsphere make_nsphere(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nsphere(param);
}
inline NPY_API nsphere* make_nsphere_ptr(const nvec4& param)
{
    nsphere* n = new nsphere ; 
    nnode::Init(*n,CSG_SPHERE) ;
    init_nsphere(*n, param);
    return n ;
}
inline NPY_API nsphere* make_nsphere_ptr(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_nsphere_ptr(param);
}


