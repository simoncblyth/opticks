#pragma once
/**
U4.hh
======

Genstep Collection
--------------------

Genstep collection within  Geant4 Scintillation+Cerenkov processes 
is central to Opticks operation, as the gensteps are the parameters 
that allow photons to be generated on GPU.  

Optical Photon Labelling
----------------------------

In pure Opticks running (junosw/opticksMode:1) there is no Geant4 generation loop, 
only in validation running (junosw/opticksMode:3) where both CPU and GPU propagations are done 
and instrumented Geant4 running (junosw/opticksMode:2) 
does generation loop monitoring become relevant and useful. 
Geant4 generation loop photon labelling is done using the API::

   U4::GenPhotonAncestor
   U4::GenPhotonBegin
   U4::GenPhotonEnd

With Geant4 generation loop monitoring every optical photon G4Track gets an spho.h label that 
is stored using G4VUserTrackInformation, via U4PhotonInfo. 
The label identifies exactly the originating photon and genstep 
and how many reemission generations have been undergone. 

Note about opticksMode
-------------------------

There is no Opticks API accepting an opticksMode argument, 
nevetheless integrations of Opticks with detector simulation 
frameworks will often find it useful to implement an opticksMode
in order to assist with controlling Opticks and comparing 
it with Geant4.  

Photon labels
---------------

TODO: check junosw/opticksMode:1 running, are the labels provided ?
 
Example of labels with {gs,ix,id,gx} ::

   In [1]: f30h
    Out[1]: 
    array([[  0,  15,  15,   0],
           [  1,  22,  56,   2],
           [  1,  10,  44,   1],
           [  1,   8,  42,   1],
           [  1,   3,  37,   0],
           [  1,   2,  36,   1],
           [  2,   3,  60,   2],
           [  3,  41, 102,   1],
           [  3,  36,  97,   2],
           [  3,  29,  90,   1],
           [  3,  19,  80,   1],
           [  4,  26, 157,   4],
           [  5,   7, 174,   1],
           [  6,   2, 195,   1]], dtype=int32)

spho.h::
                
    struct spho
    {
        int gs ; // 0-based genstep index within the event
        int ix ; // 0-based photon index within the genstep
        int id ; // 0-based photon identity index within the event 
        int gn ; // 0-based reemission index incremented at each reemission 
    ...
    };


Implementation Notes
-------------------------

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

    // genstep collection
    static const char* CollectGenstep_DsG4Scintillation_r4695_DISABLE ; 
    static const char* CollectGenstep_DsG4Scintillation_r4695_ZEROPHO ; 
    static void CollectGenstep_DsG4Scintillation_r4695( 
         const G4Track* aTrack,
         const G4Step* aStep,
         G4int    numPhotons,
         G4int    scnt,        
         G4double ScintillationTime
    ); 

    static const char* CollectGenstep_G4Cerenkov_modified_DISABLE ; 
    static const char* CollectGenstep_G4Cerenkov_modified_ZEROPHO ; 
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

    // optical photon labelling 
    static void GenPhotonAncestor(const G4Track* aTrack );                    // prior to photon generation loop(s)
    static void GenPhotonBegin( int genloop_idx );                            // start of generation loop
    static void GenPhotonEnd(   int genloop_idx, G4Track* aSecondaryTrack );  // end of generation loop

    // other 
    static NP* CollectOpticalSecondaries(const G4VParticleChange* pc ); 

    // put back old unused API, until can get integration changes thru into JUNOSW 
    static void GenPhotonSecondaries( const G4Track* , const G4VParticleChange* ); 


};


