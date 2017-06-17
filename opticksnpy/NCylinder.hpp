#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"
#include "NCylinder.h"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API ncylinder : nnode 
{

    float operator()(float x, float y, float z) const ;
    nbbox bbox() const;

    npart part();

    glm::vec3 gseedcenter() const ;
    glm::vec3 gseeddir() ;
    void pdump(const char* msg="ncylinder::pdump") const ;
 
    glm::vec3 center ; 
    float     radius ; 
    float     z1 ; 
    float     z2 ; 
};


inline NPY_API void init_cylinder(ncylinder& n, const nquad& param, const nquad& param1 )
{
    n.param = param ; 
    n.param1 = param1 ;

    n.center.x = param.f.x ; 
    n.center.y = param.f.y ; 
    n.center.z = 0.f ;

    n.radius   = param.f.w ;  
    n.z1       = param1.f.x ; 
    n.z2       = param1.f.y ; 

    assert( n.z2 > n.z1 );

    // cylinder axis in Z direction 
    //
    //      QCAP (higher Z) at   z2
    //      PCAP (lower Z)  at   z1
    //      

}

inline NPY_API ncylinder make_cylinder(const nquad& param, const nquad& param1 )
{
    ncylinder n ; 
    nnode::Init(n,CSG_CYLINDER) ; 
    init_cylinder(n, param, param1);
    return n ; 
}

inline NPY_API ncylinder make_cylinder(float radius, float z1, float z2)
{
    nquad param, param1 ;

    param.f = {0,0,0,radius} ;

    param1.f.x = z1 ; 
    param1.f.y = z2 ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_cylinder(param, param1 );
}


