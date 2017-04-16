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

inline NPY_API void init_nslab(nslab& slab, const nvec4& param, const nvec4& param1 )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.x, param.y, param.z));

    slab.n = n ; 
    slab.a = param1.x ; 
    slab.b = param1.y ; 

    slab.param.x = n.x ; 
    slab.param.y = n.y ; 
    slab.param.z = n.z ; 
    slab.param.w = 0.f ;
 
    slab.param1 = param1 ; 

    assert(slab.b > slab.a );
}
inline NPY_API nslab make_nslab(const nvec4& param, const nvec4& param1)
{
    nslab slab ; 
    nnode::Init(slab,CSG_SLAB) ; 
    init_nslab(slab, param, param1 );
    return slab ;
}

inline NPY_API nslab make_nslab(float x, float y, float z, float a, float b)
{
    nvec4 param = {x,y,z,0} ;
    nvec4 param1 = {a,b,0,0} ;
    return make_nslab( param, param1 ); 
}


inline NPY_API nslab* make_nslab_ptr(const nvec4& param, const nvec4& param1)
{
    nslab* slab = new nslab ; 
    nnode::Init(*slab,CSG_SLAB) ; 
    init_nslab(*slab, param, param1 );
    return slab ; 
}
inline NPY_API nslab* make_nslab_ptr(float x, float y, float z, float a, float b)
{
    nvec4 param = {x,y,z,0} ;
    nvec4 param1 = {a,b,0,0} ;
    return make_nslab_ptr( param, param1 ); 
}


