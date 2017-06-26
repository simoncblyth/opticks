#pragma once

#include "NNode.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API ndisc : nnode 
{
    float operator()(float x, float y, float z) const ;
    nbbox bbox() const;

    void increase_z2(float dz);
    void decrease_z1(float dz);

    glm::vec3 gseedcenter() const ;
    glm::vec3 gseeddir() const  ;
    void pdump(const char* msg="ndisc::pdump") const ;
 
    float     x() const  ; 
    float     y() const  ; 
    float     z() const  ; 
    glm::vec3 center() const  ; 
    float     radius() const  ; 
    float     z1() const  ; 
    float     z2() const  ; 
    float     r1() const  ;   // see NNodeUncoincide
    float     r2() const  ; 
};



inline NPY_API float ndisc::x() const { return param.f.x ; }
inline NPY_API float ndisc::y() const { return param.f.y ; }
inline NPY_API float ndisc::z() const { return 0.f ; }
inline NPY_API glm::vec3 ndisc::center() const { return glm::vec3(x(), y(), z()) ; }

inline NPY_API float ndisc::radius() const { return param.f.w ; }
inline NPY_API float ndisc::r1() const {     return param.f.w ; }
inline NPY_API float ndisc::r2() const {     return param.f.w ; }

inline NPY_API float ndisc::z1() const { return param1.f.x ; }
inline NPY_API float ndisc::z2() const { return param1.f.y ; }

// grow the disc upwards on upper side (z2) or downwards on down side (z1)
inline NPY_API void  ndisc::increase_z2(float dz){ assert( dz >= 0.f) ; param1.f.y += dz ; } // z2 > z1
inline NPY_API void  ndisc::decrease_z1(float dz){ assert( dz >= 0.f) ; param1.f.x -= dz ; }



inline NPY_API void init_disc(ndisc& n, const nquad& param, const nquad& param1 )
{
    n.param = param ; 
    n.param1 = param1 ;
    assert( n.z2() > n.z1() );
}

inline NPY_API ndisc make_disc(const nquad& param, const nquad& param1 )
{
    ndisc n ; 
    nnode::Init(n,CSG_DISC) ; 
    init_disc(n, param, param1);
    return n ; 
}

inline NPY_API ndisc make_disc(float radius_, float z1_, float z2_)
{
    nquad param, param1 ;

    param.f = {0,0,0,radius_} ;

    param1.f.x = z1_ ; 
    param1.f.y = z2_ ; 
    param1.u.z = 0u ; 
    param1.u.w = 0u ; 

    return make_disc(param, param1 );
}


