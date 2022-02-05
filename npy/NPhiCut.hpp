#pragma once

#include "NNode.hpp"
#include "NPY_API_EXPORT.hh"
struct nbbox ; 

/*

nphicut
=============

*/

struct nmat4triple ; 

struct NPY_API nphicut : nnode 
{
    nbbox bbox() const ; 
    float operator()(float x_, float y_, float z_) const ; 

    glm::vec3 normal(int idx) const ; 

    // placeholder zeros
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};


inline NPY_API glm::vec3 nphicut::normal(int idx) const 
{ 
    const float& cosPhi0 = param.f.x ; 
    const float& sinPhi0 = param.f.y ; 
    const float& cosPhi1 = param.f.z ; 
    const float& sinPhi1 = param.f.w ; 

    return glm::vec3( idx == 0 ? sinPhi0 : -sinPhi1 , idx == 0 ? -cosPhi0 : cosPhi1 , 0.f); 
}

inline NPY_API nphicut* make_phicut(OpticksCSG_t type )
{
    nphicut* n = new nphicut ; 
    assert( type == CSG_PHICUT || type == CSG_LPHICUT ); 
    nnode::Init(n,type) ; 
    return n ; 
}

inline NPY_API nphicut* make_phicut(OpticksCSG_t type, const nquad& param)
{
    nphicut* n = make_phicut(type); 
    n->param = param ;    
    return n ; 
}

