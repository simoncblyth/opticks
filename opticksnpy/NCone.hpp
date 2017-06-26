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
    //npart part() const ;

    glm::vec3 gseedcenter() const  ;
    glm::vec3 gseeddir() const ;
    void pdump(const char* msg="ncone::pdump") const ;
 
    glm::vec3 center() const  ; 
    glm::vec2 cnormal() const  ; 
    glm::vec2 csurface() const  ; 

    void increase_z2(float dz);
    void decrease_z1(float dz);

    float r1() const ; 
    float z1() const ; 
    float r2() const  ; 
    float z2() const  ; 

    float rmax() const  ; 
    float zc() const  ; 
    float z0() const  ; // apex
    float tantheta() const  ; 
};



inline NPY_API float ncone::r1() const { return param.f.x ; }
inline NPY_API float ncone::z1() const { return param.f.y ; }
inline NPY_API float ncone::r2() const { return param.f.z ; }
inline NPY_API float ncone::z2() const { return param.f.w ; }  // z2 > z1

// grow the cone on upwards on upper side (z2) or downwards on down side (z1)
inline NPY_API void  ncone::increase_z2(float dz){ assert( dz >= 0.f) ; param.f.w += dz ; } // z2 > z1
inline NPY_API void  ncone::decrease_z1(float dz){ assert( dz >= 0.f) ; param.f.y -= dz ; }

inline NPY_API float ncone::zc() const { return (z1() + z2())/2.f ; }
inline NPY_API float ncone::rmax() const { return fmaxf( r1(), r2())  ; }
inline NPY_API float ncone::z0() const {  return (z2()*r1()-z1()*r2())/(r1()-r2()) ; }
inline NPY_API float ncone::tantheta() const { return (r2()-r1())/(z2()-z1()) ; }
inline NPY_API glm::vec3 ncone::center() const { return glm::vec3(0.f,0.f,zc()) ; } 
inline NPY_API glm::vec2 ncone::cnormal() const { return glm::normalize( glm::vec2(z2()-z1(),r1()-r2()) ) ; }
inline NPY_API glm::vec2 ncone::csurface() const { glm::vec2 cn = cnormal() ; return glm::vec2( cn.y, -cn.x ) ; }      


inline NPY_API void init_cone(ncone& n, const nquad& param)
{
    n.param = param ;
    assert( n.z2() > n.z1() );
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


