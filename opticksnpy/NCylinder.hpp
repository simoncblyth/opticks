#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API ncylinder : nnode {

    float operator()(float x, float y, float z) ;
    nbbox bbox();

    npart part();

    glm::vec3 gcenter() ;
    void pdump(const char* msg="ncylinder::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    float     radius ; 
    float     sizeZ ; 
    unsigned  flags ; 
};


inline NPY_API void init_ncylinder(ncylinder& n, const nquad& param, const nquad& param1 )
{
    n.param = param ; 
    n.param1 = param1 ;

    n.center.x = param.f.x ; 
    n.center.y = param.f.y ; 
    n.center.z = param.f.z ;

    n.radius   = param.f.w ;  
    n.sizeZ    = param1.f.x ; 
    n.flags    = param1.u.y ; 
}

inline NPY_API ncylinder make_ncylinder(const nquad& param, const nquad& param1 )
{
    ncylinder n ; 
    nnode::Init(n,CSG_CYLINDER) ; 
    init_ncylinder(n, param, param1);
    return n ; 
}

inline NPY_API ncylinder make_ncylinder(float x, float y, float z, float radius, float sizeZ, unsigned flags)
{
    nquad param, param1 ;

    param.f = {x,y,z,radius} ;

    param1.f.x = sizeZ ; 
    param1.u.y = flags ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_ncylinder(param, param1 );
}



