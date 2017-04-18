#pragma once

#include <cassert>
#include "NGLM.hpp"
#include "NNode.hpp"

#include "NPY_API_EXPORT.hh"

/*

RTCD 

p116: Kay-Kajiya slab based volumes
p126: Closest point on plane to a point in space

*/

struct NPY_API nslab : nnode 
{
    float operator()(float x, float y, float z) ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect );

    glm::vec3 gcenter();
    void pdump(const char* msg="nslab::pdump", int verbosity=1);

    glm::vec3 n ;  // normalized normal direction
    float a ;      // signed distance from origin to plane a, 
    float b ;      // signed distance from origin to plane b, requirement b > a  (used for picking normal direction of intersects)

};

inline NPY_API void init_slab(nslab& slab, const nquad& param, const nquad& param1 )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.f.x, param.f.y, param.f.z));

    slab.n = n ; 
    slab.a = param1.f.x ; 
    slab.b = param1.f.y ; 

    slab.param.f.x = n.x ; 
    slab.param.f.y = n.y ; 
    slab.param.f.z = n.z ; 
    slab.param.f.w = 0.f ;
 
    slab.param1 = param1 ; 

    assert(slab.b > slab.a );
}
inline NPY_API nslab make_slab(const nquad& param, const nquad& param1)
{
    nslab slab ; 
    nnode::Init(slab,CSG_SLAB) ; 
    init_slab(slab, param, param1 );
    return slab ;
}

inline NPY_API nslab make_slab(float x, float y, float z, float a, float b)
{
    nquad param, param1 ; 
    param.f = {x,y,z,0} ;
    param1.f = {a,b,0,0} ;
    return make_slab( param, param1 ); 
}


