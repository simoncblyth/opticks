#pragma once

#include <CLHEP/Units/PhysicalConstants.h>

#include "globals.hh"
#include "Randomize.hh"
#include "G4ThreeVector.hh"

#ifdef DEBUG_TAG
#include "SEvt.hh"
#include "U4Stack.h"
#endif


// G.Marsaglia (1972) method
inline G4ThreeVector U4RandomDirection()
{
  G4double u0, u1 ;  

  G4double u, v, b;
  do {

    u0 = G4UniformRand() ;
#ifndef PRODUCTION
#ifdef DEBUG_TAG
    SEvt::AddTag(1, U4Stack_RandomDirection, u0 ); 
#endif
#endif
    u1 = G4UniformRand() ;
#ifndef PRODUCTION
#ifdef DEBUG_TAG
    SEvt::AddTag(1, U4Stack_RandomDirection, u1 ); 
#endif
#endif
    u = 2.*u0 - 1.; 
    v = 2.*u1 - 1.; 
    b = u*u + v*v;
  } while (b > 1.);
  G4double a = 2.*std::sqrt(1. - b); 
  return G4ThreeVector(a*u, a*v, 2.*b - 1.);
}


