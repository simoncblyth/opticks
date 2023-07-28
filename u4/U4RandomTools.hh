#pragma once

#include <CLHEP/Units/PhysicalConstants.h>

#include "globals.hh"
#include "Randomize.hh"
#include "G4TwoVector.hh"
#include "G4ThreeVector.hh"
#include "U4RandomDirection.hh"

#ifdef DEBUG_TAG
#include "U4Stack.h"
#include "SEvt.hh"
#endif



// ---------------------------------------------------------------------------
// Returns a random lambertian unit vector (rejection sampling)
//
inline G4ThreeVector U4LambertianRand(const G4ThreeVector& normal)
{
  G4ThreeVector vect;
  G4double ndotv;
  G4int count=0;
  const G4int max_trials = 1024;

  G4double u_exitloop ;  

  do  
  {
    ++count;
    vect = U4RandomDirection();
    ndotv = normal * vect;

    if (ndotv < 0.0)
    {   
      vect = -vect;
      ndotv = -ndotv;
    }   

    u_exitloop = G4UniformRand() ;
#ifdef DEBUG_TAG
    SEvt::AddTag(1, U4Stack_LambertianRand, u_exitloop ); 
#endif  

  } while (!(u_exitloop < ndotv) && (count < max_trials));

  return vect;
}


