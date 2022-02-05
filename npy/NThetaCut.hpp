#pragma once

#include "NNode.hpp"
#include "NPY_API_EXPORT.hh"
struct nbbox ; 

/*

nthetacut
=============

*/

struct nmat4triple ; 

struct NPY_API nthetacut : nnode 
{
    float operator()(float x_, float y_, float z_) const ; 


    // placeholders : without then get NCSG::Adopt assert with X4PhysicalVolume::ConvertSolid_FromRawNode
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};


inline NPY_API nthetacut* make_thetacut(OpticksCSG_t type )
{
    nthetacut* n = new nthetacut ; 
    assert( type == CSG_THETACUT || type == CSG_LTHETACUT ); 
    nnode::Init(n,type) ; 
    return n ; 
}

inline NPY_API nthetacut* make_thetacut(OpticksCSG_t type, const nquad& param)
{
    nthetacut* n = make_thetacut(type); 
    n->param = param ;    
    return n ; 
}

inline NPY_API nthetacut* make_thetacut(OpticksCSG_t type, double theta0_pi, double theta1_pi )
{
    const double pi = glm::pi<double>() ; 

    double theta0 = theta0_pi*pi ; 
    double theta1 = theta1_pi*pi ; 

    double cosTheta0 = std::cos(theta0);
    double cosTheta0Sign = cosTheta0/std::abs(cosTheta0) ; 
    double tanTheta0 = std::tan(theta0); 
    double tan2Theta0 = tanTheta0*tanTheta0 ; 

    double cosTheta1 = std::cos(theta1); 
    double cosTheta1Sign = cosTheta1/std::abs(cosTheta1) ; 
    double tanTheta1 = std::tan(theta1); 
    double tan2Theta1 = tanTheta1*tanTheta1 ; 
  
    // following Lucas recommendations from CSG/csg_intersect_leaf_thetacut.h 
    nquad param ; 
    param.f.x = float(theta0_pi == 0.5 ? 0. : cosTheta0Sign ); 
    param.f.y = float(theta0_pi == 0.5 ? 0. : tan2Theta0 ); 
    param.f.z = float(theta1_pi == 0.5 ? 0. : cosTheta1Sign ); 
    param.f.w = float(theta1_pi == 0.5 ? 0. : tan2Theta1 ); 

    nthetacut* n = make_thetacut(type, param); 
    return n ; 
}


