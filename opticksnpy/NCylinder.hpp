#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"
#include "NCylinder.h"

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


inline NPY_API void init_cylinder(ncylinder& n, const nquad& param, const nquad& param1 )
{
    n.param = param ; 
    n.param1 = param1 ;

    n.center.x = param.f.x ; 
    n.center.y = param.f.y ; 
    n.center.z = param.f.z ;

    n.radius   = param.f.w ;  
    n.sizeZ    = param1.f.x ; 
    n.flags    = param1.u.y ; 

    // cylinder axis in Z direction 
    //
    //      QCAP (higher Z) at   center.z + sizeZ/2
    //      PCAP (lower Z)  at   center.z - sizeZ/2
    //      

}

inline NPY_API ncylinder make_cylinder(const nquad& param, const nquad& param1 )
{
    ncylinder n ; 
    nnode::Init(n,CSG_CYLINDER) ; 
    init_cylinder(n, param, param1);
    return n ; 
}

inline NPY_API ncylinder make_cylinder(float x, float y, float z, float radius, float sizeZ, unsigned flags)
{
    nquad param, param1 ;

    param.f = {x,y,z,radius} ;

    param1.f.x = sizeZ ; 
    param1.u.y = flags ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_cylinder(param, param1 );
}



