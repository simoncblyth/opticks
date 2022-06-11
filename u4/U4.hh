#pragma once
/**
U4.hh
======

Note that Opticks types are mostly kept out of this header in order to simplify 
usage from detector framework code.  For example this is done by:

1. using private methods that create the Opticks types
2. retaining pointers to results in standard places elsewhere, mostly in SEvt, 
   rather than directly returning them. 

**/

#include "plog/Severity.h"

struct NP ; 
class G4VParticleChange ; 
class G4Track ; 
class G4Step ; 

#include "G4Types.hh"
#include "U4_API_EXPORT.hh"

struct U4_API U4
{
    static const plog::Severity LEVEL ;

    static void CollectGenstep_DsG4Scintillation_r4695( 
         const G4Track* aTrack,
         const G4Step* aStep,
         G4int    numPhotons,
         G4int    scnt,        
         G4double ScintillationTime
    ); 

    static void CollectGenstep_G4Cerenkov_modified( 
        const G4Track* aTrack,
        const G4Step* aStep,
        G4int    numPhotons,
        G4double    betaInverse,
        G4double    pmin,
        G4double    pmax,
        G4double    maxCos,
        G4double    maxSin2,
        G4double    meanNumberOfPhotons1,
        G4double    meanNumberOfPhotons2
    );

    static void GenPhotonAncestor(const G4Track* aTrack );                    // prior to photon generation loop(s)
    static void GenPhotonBegin( int genloop_idx );                            // start of generation loop
    static void GenPhotonEnd(   int genloop_idx, G4Track* aSecondaryTrack );  // end of generation loop
    static void GenPhotonSecondaries(const G4Track* aTrack, const G4VParticleChange* change ); 
    static NP* CollectOpticalSecondaries(const G4VParticleChange* pc ); 
};


