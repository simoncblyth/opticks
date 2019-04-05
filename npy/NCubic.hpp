#pragma once

/*
https://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/hyperboloid.html
*/

#include "NNode.hpp"
#include "NQuad.hpp"
#include "NGLM.hpp"

struct npart ;
struct nbbox ; 
struct nuv ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API ncubic : nnode 
{
    float operator()(float x, float y, float z) const ;

    const float& A() const ;
    const float& B() const ;
    const float& C() const ;
    const float& D() const ;
    float z1() const ;
    float z2() const ;

    float rrmax() const ;
    float rrz(float z) const ;
    float  rz(float z) const ;

    nbbox bbox() const ;

    unsigned  par_nsurf() const ; 
    glm::vec3 par_pos_model(const nuv& uv) const  ;
    int       par_euler() const ; 
    unsigned  par_nvertices(unsigned nu, unsigned nv) const ; 

    static void _par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_ ) ;

    void pdump(const char* msg="ncubic::pdump") const ;
};

inline NPY_API const float& ncubic::A() const { return param.f.x ; }
inline NPY_API const float& ncubic::B() const { return param.f.y ; }
inline NPY_API const float& ncubic::C() const { return param.f.z ; }
inline NPY_API const float& ncubic::D() const { return param.f.w ; }
inline NPY_API float ncubic::z1() const { return param1.f.x ; }
inline NPY_API float ncubic::z2() const { return param1.f.y ; }

inline NPY_API float ncubic::rrz(float z) const { return (((A()*z+B())*z)+C())*z + D()  ; }
inline NPY_API float ncubic::rz(float z) const { return sqrt(rrz(z)) ;  }


inline NPY_API void init_cubic(ncubic* n, const nquad& param, const nquad& param1)
{
    n->param = param ; 
    n->param1 = param1 ; 
}
inline NPY_API ncubic* make_cubic(const nquad& param, const nquad& param1 )
{
    ncubic* n = new ncubic ; 
    nnode::Init(n,CSG_CUBIC) ; 
    init_cubic(n, param, param1);
    return n ; 
}
inline NPY_API ncubic* make_cubic(float A=100.f, float B=100.f, float C=100.f, float D=100.f, float z1=-100.f, float z2=100.f, float zspare1=0.f, float zspare2=0.f)
{
    nquad param, param1 ; 
    param.f = {A,B,C,D} ;
    param1.f = {z1,z2,zspare1,zspare2} ;
    return make_cubic(param, param1);
}


