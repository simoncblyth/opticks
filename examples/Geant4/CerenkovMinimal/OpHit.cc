#include "OpHit.hh"

OpHit* OpHit::MakeDummyHit()
{  
    OpHit* hit = new OpHit ; 
    hit->ene = 1. ; 
    hit->tim = 1. ; 
    hit->pos = G4ThreeVector(1,1,1); 
    hit->dir = G4ThreeVector(0,0,1); 
    hit->pol = G4ThreeVector(1,0,0); 

    return hit ; 
}


