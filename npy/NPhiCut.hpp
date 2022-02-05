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
    float operator()(float x_, float y_, float z_) const ; 

    glm::vec3 normal(int idx) const ; 

    // placeholder : otherwise X4PhysicalVolume::ConvertSolid_FromRawNode asserts in NCSG::Adopt nnode::collectParPoints
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

inline NPY_API nphicut* make_phicut(OpticksCSG_t type, double startPhi_pi, double deltaPhi_pi )
{
    double phi0 = startPhi_pi ; 
    double phi1 = startPhi_pi + deltaPhi_pi ;

    const double pi = glm::pi<double>() ; 
    double cosPhi0 = std::cos(phi0*pi) ;
    double sinPhi0 = std::sin(phi0*pi) ;
    double cosPhi1 = std::cos(phi1*pi) ;
    double sinPhi1 = std::sin(phi1*pi) ;

    nquad param ; 
    param.f.x = float(cosPhi0); 
    param.f.y = float(sinPhi0); 
    param.f.z = float(cosPhi1); 
    param.f.w = float(sinPhi1); 

    nphicut* n = make_phicut(type, param); 
    return n ; 
}




