#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API ndisc : nnode 
{

    float operator()(float x, float y, float z) const ;
    nbbox bbox() const;

    npart part();

    glm::vec3 gseedcenter() const ;
    glm::vec3 gseeddir() ;
    void pdump(const char* msg="ndisc::pdump") const ;
 
    glm::vec3 center ; 
    float     radius ; 
    float     z1 ; 
    float     z2 ; 
};


inline NPY_API void init_disc(ndisc& n, const nquad& param, const nquad& param1 )
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

}

inline NPY_API ndisc make_disc(const nquad& param, const nquad& param1 )
{
    ndisc n ; 
    nnode::Init(n,CSG_DISC) ; 
    init_disc(n, param, param1);
    return n ; 
}

inline NPY_API ndisc make_disc(float radius, float z1, float z2)
{
    nquad param, param1 ;

    param.f = {0,0,0,radius} ;

    param1.f.x = z1 ; 
    param1.f.y = z2 ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_disc(param, param1 );
}


