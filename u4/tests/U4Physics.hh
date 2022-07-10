#pragma once
/**
tests/U4Physics.hh : Not remotely reusable so consigned to tests folder together with DsG4Scintillation
=========================================================================================================

This is intended solely for use from U4RecorderTest 

**/

#include <cstdlib>
#include <string>
#include <sstream>
#include "G4VUserPhysicsList.hh"

class G4Cerenkov_modified ; 
class DsG4Scintillation ; 

#ifdef DEBUG_TAG
class ShimG4OpAbsorption ;
class ShimG4OpRayleigh ;
#else
class G4OpAbsorption ;
class G4OpRayleigh ;
#endif

class InstrumentedG4OpBoundaryProcess ; 


struct U4Physics : public G4VUserPhysicsList
{
    static int EInt(const char* key, const char* fallback="0"); 

    G4Cerenkov_modified*  fCerenkov ; 
    DsG4Scintillation*    fScintillation ; 

#ifdef DEBUG_TAG
    ShimG4OpAbsorption*   fAbsorption ;
    ShimG4OpRayleigh*     fRayleigh ;
#else
    G4OpAbsorption*       fAbsorption ;
    G4OpRayleigh*         fRayleigh ;
#endif

    InstrumentedG4OpBoundaryProcess*  fBoundary ;

    static std::string Desc(); 
    U4Physics(); 

    void ConstructParticle();
    void ConstructProcess();
    void ConstructEM();
    void ConstructOp();
};



