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

    float r1() const { return param.f.x ; }
    float z1() const { return param.f.y ; }
    float r2() const { return param.f.z ; }
    float z2() const { return param.f.w ; }
    float zc() const { return (z1() + z2())/2.f ; }
    float rmax() const {  return fmaxf( r1(), r2() ) ; }

    glm::vec3 gseedcenter() ;
    glm::vec3 gseeddir() ;
    void pdump(const char* msg="ncone::pdump", int verbosity=1);
 
    glm::vec3 center ; 
};


inline NPY_API void init_cone(ncone& n, const nquad& param)
{
    n.param = param ; 
    n.center.x = 0.f ; 
    n.center.y = 0.f ; 
    n.center.z = n.zc() ;
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


