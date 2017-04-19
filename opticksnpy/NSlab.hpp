#pragma once

#include <cassert>
#include "NGLM.hpp"
#include "NNode.hpp"
#include "NSlab.h"

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

    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();
    void pdump(const char* msg="nslab::pdump", int verbosity=1);

    glm::vec3 n ;  // normalized normal direction
    float a ;      // signed distance from origin to plane a, 
    float b ;      // signed distance from origin to plane b, requirement b > a  (used for picking normal direction of intersects)

    unsigned flags ;  // eg 3 for:  SLAB_ACAP|SLAB_BCAP

};

inline NPY_API void init_slab(nslab& slab, const nquad& param, const nquad& param1 )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.f.x, param.f.y, param.f.z));
    slab.flags = param.u.w ;  

    slab.n = n ; 
    slab.a = param1.f.x ; 
    slab.b = param1.f.y ; 

    slab.param.f.x = n.x ; 
    slab.param.f.y = n.y ; 
    slab.param.f.z = n.z ; 
    slab.param.u.w = slab.flags ;
 
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

inline NPY_API nslab make_slab(float x, float y, float z, float a, float b, unsigned flags=SLAB_ACAP|SLAB_BCAP)
{
    nquad param, param1 ; 

    param.f.x = x ;
    param.f.y = y ;
    param.f.z = z ;
    param.u.w = flags ; 

    param1.f.x = a ;
    param1.f.y = b ;

    return make_slab( param, param1 ); 
}


