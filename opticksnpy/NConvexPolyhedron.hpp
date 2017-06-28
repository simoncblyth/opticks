#pragma once

#include <cassert>
#include "NGLM.hpp"
#include "NNode.hpp"

#include "NPY_API_EXPORT.hh"

struct NPY_API nconvexpolyhedron : nnode 
{
    float operator()(float x, float y, float z) const ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect ) const ;

    nbbox bbox() const ;
    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();

    glm::vec3 par_pos_model(const nuv& uv) const  ;
    unsigned  par_nsurf() const ; 
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 


    void pdump(const char* msg="nconvexpolyhedron::pdump") const ;
};

inline NPY_API void init_convexpolyhedron(nconvexpolyhedron& cpol, const nquad& param, const nquad& param1, const nquad& param2, const nquad& param3 )
{
    cpol.param = param ; 
    cpol.param1 = param1 ; 
    cpol.param2 = param2 ; 
    cpol.param3 = param3 ; 
}

inline NPY_API nconvexpolyhedron make_convexpolyhedron(const nquad& param, const nquad& param1, const nquad& param2, const nquad& param3)
{
    nconvexpolyhedron cpol ; 
    nnode::Init(cpol,CSG_CONVEXPOLYHEDRON) ; 
    init_convexpolyhedron(cpol, param, param1, param2, param3 );
    return cpol ;
}


