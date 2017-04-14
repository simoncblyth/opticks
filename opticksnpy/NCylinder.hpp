#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API ncylinder : nnode {

    // NO CTOR

    double operator()(double px, double py, double pz) ;
    nbbox bbox();

    npart part();

    glm::vec3 gcenter() ;
    void pdump(const char* msg="ncylinder::pdump", int verbosity=1);
 
    glm::vec3 center ; 
    float     radius_ ; 

};


inline NPY_API void init_ncylinder(ncylinder& n, const nvec4& param)
{
    n.param.x = param.x ; 
    n.param.y = param.y ; 
    n.param.z = param.z ; 
    n.param.w = param.w ; 

    n.center.x = param.x ; 
    n.center.y = param.y ; 
    n.center.z = param.z ;
    n.radius_  = param.w ;  
}

inline NPY_API ncylinder make_ncylinder(const nvec4& param)
{
    ncylinder n ; 
    nnode::Init(n,CSG_CYLINDER) ; 
    init_ncylinder(n, param);
    return n ; 
}
inline NPY_API ncylinder make_ncylinder(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_ncylinder(param);
}
inline NPY_API ncylinder* make_ncylinder_ptr(const nvec4& param)
{
    ncylinder* n = new ncylinder ; 
    nnode::Init(*n,CSG_CYLINDER) ;
    init_ncylinder(*n, param);
    return n ;
}
inline NPY_API ncylinder* make_ncylinder_ptr(float x, float y, float z, float w)
{
    nvec4 param = {x,y,z,w} ;
    return make_ncylinder_ptr(param);
}


