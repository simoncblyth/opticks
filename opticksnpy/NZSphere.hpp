#pragma once

#include <cassert>

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

struct nbbox ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API nzsphere : nnode {

    float operator()(float x, float y, float z) ;

    nbbox bbox();

    glm::vec3 gcenter() ;
    void pdump(const char* msg="nzsphere::pdump", int verbosity=1);

    float zmax(){ return center.z + zdelta.y ; }
    float zmin(){ return center.z + zdelta.x ; }
 
    glm::vec3 center ; 
    glm::vec2 zdelta ;   // z range relative to center.z, ie -radius, radius would correspond to full sphere
    float     radius ; 

};

inline NPY_API void init_zsphere(nzsphere& s, const nquad& param, const nquad& param1)
{
    s.param = param ; 
    s.param1 = param1 ; 

    s.center.x = param.f.x ; 
    s.center.y = param.f.y ; 
    s.center.z = param.f.z ;
    s.radius   = param.f.w ;  

    s.zdelta.x = param1.f.x  ;
    s.zdelta.y = param1.f.y  ;

    assert( s.zdelta.y > s.zdelta.x );
}

inline NPY_API nzsphere make_zsphere(const nquad& param, const nquad& param1)
{
    nzsphere n ; 
    nnode::Init(n,CSG_ZSPHERE) ; 
    init_zsphere(n, param, param1 );
    return n ; 
}

inline NPY_API nzsphere make_zsphere(float x, float y, float z, float radius, float zdelta_min, float zdelta_max)
{
    nquad param, param1 ; 
    param.f = {x,y,z,radius} ;
    param1.f = {zdelta_min, zdelta_max, 0,0} ;
    return make_zsphere(param, param1);
}

