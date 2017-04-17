#pragma once

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

struct nplane ; 
struct ndisc ; 
struct npart ;
struct nbbox ; 


#include "NPY_API_EXPORT.hh"

struct NPY_API nsphere : nnode {

    // NO CTOR

    float costheta(float z);

    float operator()(float x, float y, float z) ;

    nbbox bbox();

    npart part();
    static ndisc intersect(nsphere& a, nsphere& b);

    // result of intersect allows partitioning 
    npart zrhs(const ndisc& dsc); // +z to the right  
    npart zlhs(const ndisc& dsc);  

    glm::vec3 gcenter() ;
    void pdump(const char* msg="nsphere::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    float     radius ; 

};



inline NPY_API void init_nsphere(nsphere& s, const nquad& param)
{
    s.param = param ; 

    s.center.x = param.f.x ; 
    s.center.y = param.f.y ; 
    s.center.z = param.f.z ;
    s.radius  = param.f.w ;  
}

inline NPY_API nsphere make_nsphere(const nquad& param)
{
    nsphere n ; 
    nnode::Init(n,CSG_SPHERE) ; 
    init_nsphere(n, param);
    return n ; 
}
inline NPY_API nsphere make_nsphere(float x, float y, float z, float w)
{
    nquad param ; 
    param.f = {x,y,z,w} ;
    return make_nsphere(param);
}


