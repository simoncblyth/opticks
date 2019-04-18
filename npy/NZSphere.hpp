#pragma once

#include <cassert>

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"
#include "NZSphere.h"

struct nbbox ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API nzsphere : nnode {

    float operator()(float x, float y, float z) const ;

    nbbox bbox() const ;

    glm::vec3 gseedcenter() const  ;
    glm::vec3 gseeddir() const ;

    void pdump(const char* msg="nzsphere::pdump") const ;

    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_, const float z1_, const float z2_, const bool has_z1cap_ , const bool has_z2cap_  ) ;

    bool      has_z1_endcap() const ;
    bool      has_z2_endcap() const ;



    unsigned flags() const ;

    float     x() const ; 
    float     y() const ; 
    float     z() const ; 
    float     radius() const  ; 
    float     r1() const  ;    // of the endcaps at z1,z2  
    float     r2() const  ; 
    float     rz(float z) const ; // radius of z-slice endcap

    float  startTheta() const ; 
    float  endTheta() const ; 
    float  deltaTheta() const ; 

    glm::vec3 center() const  ; 

    float     z2() const ; 
    float     z1() const ; 

    float zmax() const ;
    float zmin() const ;
    float zc() const ;


    void   check() const ; 

    void increase_z2(float dz);
    void decrease_z1(float dz);
    void set_zcut(float z1, float z2);


};


inline NPY_API glm::vec3 nzsphere::center() const { return glm::vec3(x(),y(),z()) ;  } 

inline NPY_API float nzsphere::x() const {      return param.f.x ; }
inline NPY_API float nzsphere::y() const {      return param.f.y ; }
inline NPY_API float nzsphere::z() const {      return param.f.z ; }
inline NPY_API float nzsphere::radius() const { return param.f.w ; }

inline NPY_API float nzsphere::z2() const {      return param1.f.y ; }  // z2 > z1
inline NPY_API float nzsphere::z1() const {      return param1.f.x ; }
inline NPY_API float nzsphere::r1() const {      return rz(z1()) ; } 
inline NPY_API float nzsphere::r2() const {      return rz(z2()) ; } 

inline NPY_API float nzsphere::zmax() const {    return z() + z2() ; }
inline NPY_API float nzsphere::zmin() const {    return z() + z1() ; }
inline NPY_API float nzsphere::zc() const {      return (zmin() + zmax())/2.f ; }

inline NPY_API unsigned nzsphere::flags() const { return param2.u.x ; }

// grow the zsphere upwards on upper side (z2) or downwards on down side (z1)
inline NPY_API void  nzsphere::increase_z2(float dz){ assert( dz >= 0.f) ; param1.f.y += dz ; check() ; } // z2 > z1
inline NPY_API void  nzsphere::decrease_z1(float dz){ assert( dz >= 0.f) ; param1.f.x -= dz ; check() ; }
inline NPY_API void  nzsphere::set_zcut(float z1, float z2){  param1.f.x = z1 ; param1.f.y = z2 ; check() ; }

inline NPY_API void nzsphere::check() const 
{
    assert( z2() > z1() );
    assert( fabs(z2()) <= radius() );
    assert( fabs(z1()) <= radius() );
    assert( zmax() > zmin() ); 
}

inline NPY_API float nzsphere::rz(float z_) const 
{
    float r = radius(); 
    return sqrt(r*r - z_*z_) ;  
}

/*
In [23]: np.arccos([1,0,-1])/np.pi
Out[23]: array([ 0. ,  0.5,  1. ])
*/ 

inline float NPY_API nzsphere::startTheta() const { return acosf(z2()/radius()); }
inline float NPY_API nzsphere::endTheta() const {   return acosf(z1()/radius()); }
inline float NPY_API nzsphere::deltaTheta() const { return endTheta() - startTheta(); }



inline NPY_API void init_zsphere(nzsphere* s, const nquad& param, const nquad& param1, const nquad& param2)
{
    s->param = param ; 
    s->param1 = param1 ; 
    s->param2 = param2 ; 
    s->check();
}

inline NPY_API nzsphere* make_zsphere(const nquad& param, const nquad& param1, const nquad& param2)
{
    nzsphere* n = new nzsphere ; 
    nnode::Init(n,CSG_ZSPHERE) ; 
    init_zsphere(n, param, param1, param2 );
    return n ; 
}

inline NPY_API nzsphere* make_zsphere(float x, float y, float z, float radius, float z1, float z2 )
{
    nquad p0, p1, p2  ; 

    p0.f = {x,y,z,radius} ;
    p1.f = {z1, z2, 0,0} ;
    p2.u = {3, 0, 0, 0} ;   // <-- fix to avoid uninitialized gibberish getting into partBuffer

    // the 3 is to match endcap flags in an old version 

    return make_zsphere(p0, p1, p2);
}

inline NPY_API nzsphere* make_zsphere()
{
    return make_zsphere(0.f, 0.f, 0.f, 100.f, -50.f, 70.f );
}

inline NPY_API nzsphere* make_zsphere(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )
{
    // used by code generation 
    assert( z1 == 0.f );
    assert( w1 == 0.f );
    return make_zsphere( x0,y0,z0,w0,x1,y1 );
}

