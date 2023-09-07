#pragma once

#include "G4ThreeVector.hh"

struct S4OpBoundaryProcess
{
    static G4ThreeVector SmearNormal_SigmaAlpha(const G4ThreeVector& Momentum, const G4ThreeVector&  Normal, G4double sigma_alpha );  
    static G4ThreeVector SmearNormal_Polish(    const G4ThreeVector& Momentum, const G4ThreeVector&  Normal, G4double polish      );  
};


#include "Randomize.hh"
#include <CLHEP/Units/PhysicalConstants.h>

/**
S4OpBoundaryProcess::SmearNormal_SigmaAlpha
--------------------------------------------

Extract from G4OpBoundaryProcess::GetFacetNormal to investigate how to 
do the equivalent with CUDA. 

HMM: can implement all this in CUDA : the difficulty is more with 
how to implement switching to alternative "ground" rather than "polished" 
handling. Maybe best way to switch by expanding the ems enum used in qsim::propagate::

    2025     if( command == BOUNDARY )
    2026     {
    2027         const int& ems = ctx.s.optical.y ;
    ....
    2040         if( ems == smatsur_NoSurface )
    2041         {
    2042             command = propagate_at_boundary( flag, rng, ctx ) ;
    2043         }
    2044         else if( ems == smatsur_Surface )
    2045         {
    2046             command = propagate_at_surface( flag, rng, ctx ) ;
    2047         }

Or branch based on ems or some other surface type enum within::

    qsim::propagate_at_surface

* HMM: probably best to defer this until find a corresponding discrepancy ?
  So can change just the thing to get match


Q1:Where to get sigma_alpha and polish from ?
A1:U4Surface::MakeFold plants ModelValue, etc.. into surface metadata::

    288         G4double ModelValue = theModel == glisur ? os->GetPolish() : os->GetSigmaAlpha() ;
    289         assert( ModelValue >= 0. && ModelValue <= 1. );
    ...
    302         sub->set_meta<int>("Type", theType) ;
    303         sub->set_meta<int>("Model", theModel) ;
    304         sub->set_meta<int>("Finish", theFinish) ;
    305         sub->set_meta<double>("ModelValue", ModelValue ) ;

The value percentage is in optical buffer .w already  
could trivially change to setting float32 value into 
the mostly int optical buffer.:: 

    In [8]: f.bnd_names[4]
    Out[8]: 'Water/NNVTMaskOpticalSurface//CDReflectorSteel'

    In [9]: f.optical[4]
    Out[9]: 
    array([[ 3,  0,  0,  0],
           [27,  2,  3, 20],
           [ 0,  1,  0,  0],
           [ 2,  0,  0,  0]], dtype=int32)


Q2: How to switch on FacetNormal smearing with SmearNormal_SigmaAlpha/SmearNormal_Polish ? 
    How to pick between them ? 



**/

inline G4ThreeVector S4OpBoundaryProcess::SmearNormal_SigmaAlpha(const G4ThreeVector& Momentum, const G4ThreeVector&  Normal, G4double sigma_alpha )
{
    if (sigma_alpha == 0.0) return Normal;
    G4ThreeVector FacetNormal ; 

    G4double alpha;
    G4double f_max = std::min(1.0,4.*sigma_alpha);

    G4double phi, SinAlpha, CosAlpha, SinPhi, CosPhi, unit_x, unit_y, unit_z;
    G4ThreeVector tmpNormal;

    do {
       do {
          alpha = G4RandGauss::shoot(0.0,sigma_alpha);
          // Loop checking, 13-Aug-2015, Peter Gumplinger
       } while (G4UniformRand()*f_max > std::sin(alpha) || alpha >= CLHEP::halfpi );

       phi = G4UniformRand()*CLHEP::twopi;

       SinAlpha = std::sin(alpha);
       CosAlpha = std::cos(alpha);
       SinPhi = std::sin(phi);
       CosPhi = std::cos(phi);

       unit_x = SinAlpha * CosPhi;
       unit_y = SinAlpha * SinPhi;
       unit_z = CosAlpha;

       FacetNormal.setX(unit_x);
       FacetNormal.setY(unit_y);
       FacetNormal.setZ(unit_z);

       tmpNormal = Normal;  
       // HUH: argument to rotateUz and Normal are both const refs
       // so tmpNormal is entirely pointless ? 

       FacetNormal.rotateUz(tmpNormal);
       // Loop checking, 13-Aug-2015, Peter Gumplinger
    } while (Momentum * FacetNormal >= 0.0);
    return FacetNormal ; 
}

inline G4ThreeVector S4OpBoundaryProcess::SmearNormal_Polish(     const G4ThreeVector& Momentum, const G4ThreeVector&  Normal, G4double polish )
{
    if (polish == 1.0) return Normal;
    G4ThreeVector FacetNormal ; 

    do {
       G4ThreeVector smear;
       do {
          smear.setX(2.*G4UniformRand()-1.0);
          smear.setY(2.*G4UniformRand()-1.0);
          smear.setZ(2.*G4UniformRand()-1.0);
          // Loop checking, 13-Aug-2015, Peter Gumplinger
       } while (smear.mag()>1.0);
       smear = (1.-polish) * smear;
       FacetNormal = Normal + smear;
       // Loop checking, 13-Aug-2015, Peter Gumplinger
    } while (Momentum * FacetNormal >= 0.0);
    FacetNormal = FacetNormal.unit();
    return FacetNormal ; 
}

