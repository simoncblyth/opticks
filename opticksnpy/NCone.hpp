#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"
//#include "NCone.h"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API ncone : nnode {

    float operator()(float x, float y, float z) ;
    nbbox bbox();

    npart part();

    glm::vec3 gseedcenter() ;
    glm::vec3 gseeddir() ;
    void pdump(const char* msg="ncone::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    float     radius ; 
    float     sizeZ ; 
    unsigned  flags ; 
};


inline NPY_API void init_cone(ncone& n, const nquad& param, const nquad& param1 )
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

inline NPY_API ncone make_cone(const nquad& param, const nquad& param1 )
{
    ncone n ; 
    nnode::Init(n,CSG_CONE) ; 
    init_cone(n, param, param1);
    return n ; 
}

inline NPY_API ncone make_cone(float x, float y, float z, float radius, float sizeZ, unsigned flags)
{
    nquad param, param1 ;

    param.f = {x,y,z,radius} ;

    param1.f.x = sizeZ ; 
    param1.u.y = flags ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_cone(param, param1 );
}



