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
    float operator()(float x, float y, float z) const ;

    bool intersect( const float tmin, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect );

    glm::vec3 gseedcenter();
    glm::vec3 gseeddir();
    void pdump(const char* msg="nslab::pdump") const ;

    glm::vec3 center() const ;
    glm::vec3 normal() const  ;  // normalized normal direction
    float     a() const ;        // signed distance from origin to plane a, 
    float     b() const ;        // signed distance from origin to plane b, requirement b > a  (used for picking normal direction of intersects)
    unsigned flags() const ;     // now always 3: SLAB_ACAP|SLAB_BCAP


    // parametric surface positions 

    glm::vec3 par_pos_model(const nuv& uv) const ;  // no transforms, bare model
    unsigned  par_nsurf() const ;
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ;
    int       par_euler() const ; 

    void     _par_pos_plane(glm::vec3& pos, const nuv& uv) const ; // add offsets to pos that cover a region of plane


    void define_uv_basis();

    glm::vec3 udir ;
    glm::vec3 vdir ;


};



inline glm::vec3 nslab::normal() const  // normalization done at init
{
    return glm::vec3(param.f.x, param.f.y, param.f.z) ;
}
inline NPY_API float nslab::a() const 
{
    return param1.f.x ; 
}
inline NPY_API float nslab::b() const 
{
    return param1.f.y ; 
}
inline NPY_API unsigned nslab::flags() const 
{
    unsigned flags_ = param.u.w ;
    assert( flags_ == (SLAB_ACAP|SLAB_BCAP) ); 
    return flags_ ; 
}


inline NPY_API void init_slab(nslab* slab, const nquad& param, const nquad& param1 )
{
    glm::vec3 n = glm::normalize(glm::vec3(param.f.x, param.f.y, param.f.z));

    slab->param.f.x = n.x ; 
    slab->param.f.y = n.y ; 
    slab->param.f.z = n.z ; 
    slab->param.u.w = SLAB_ACAP|SLAB_BCAP ; // caps are now always ON, as makes no-sense to be off
 
    slab->param1.f.x = param1.f.x ; 
    slab->param1.f.y = param1.f.y ; 

    assert(slab->b() > slab->a() );

    slab->define_uv_basis();

}

inline NPY_API nslab* make_slab(const nquad& param, const nquad& param1)
{
    nslab* n = new nslab  ; 
    nnode::Init(n,CSG_SLAB) ; 
    init_slab(n, param, param1 );
    return n ;
}

inline NPY_API nslab* make_slab(float x, float y, float z, float a, float b  )
{
    nquad param, param1 ; 

    param.f.x = x ;
    param.f.y = y ;
    param.f.z = z ;

    param1.f.x = a ;
    param1.f.y = b ;

    return make_slab( param, param1 ); 
}

inline NPY_API nslab* make_slab(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )
{
    // 2-quad form used by codegen
    assert( w0 == 0.f );
    assert( z1 == 0.f );
    assert( w1 == 0.f );

    return make_slab(x0,y0,z0,x1,y1);
}




