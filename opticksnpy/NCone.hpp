#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API ncone : nnode 
{
    float operator()(float x, float y, float z) const ;
    nbbox bbox() const;
    npart part();

    glm::vec3 gseedcenter() ;
    glm::vec3 gseeddir() ;
    void pdump(const char* msg="ncone::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    glm::vec2 cnormal ; 
    glm::vec2 csurface ; 

    float r1 ; 
    float z1 ; 
    float r2 ; 
    float z2 ; 

    float rmax ; 
    float zc ; 
    float z0 ; // apex
    float tantheta ; 
};


inline NPY_API void init_cone(ncone& n, const nquad& param)
{
    n.param = param ;
 
    n.r1 = param.f.x ; 
    n.z1 = param.f.y ; 
    n.r2 = param.f.z ; 
    n.z2 = param.f.w ; 
   
    assert( n.z2 > n.z1 );

    n.rmax = fmaxf( n.r1, n.r2 );
    n.zc = (n.z1 + n.z2)/2.f ; 
    n.z0 = (n.z2*n.r1-n.z1*n.r2)/(n.r1-n.r2) ;
    n.tantheta = (n.r2-n.r1)/(n.z2-n.z1) ;

    n.center.x = 0.f ; 
    n.center.y = 0.f ; 
    n.center.z = n.zc ;

    n.cnormal = glm::normalize( glm::vec2(n.z2-n.z1,n.r1-n.r2)) ;     
    n.csurface = glm::vec2( n.cnormal.y, -n.cnormal.x ) ;     

}

inline NPY_API ncone make_cone(const nquad& param)
{
    ncone n ; 
    nnode::Init(n,CSG_CONE) ; 
    init_cone(n, param);
    return n ; 
}

inline NPY_API ncone make_cone(float r1, float z1, float r2, float z2)
{
    nquad param ;
    param.f = {r1,z1,r2,z2} ;
    return make_cone(param);
}


