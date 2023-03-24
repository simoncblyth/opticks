#pragma once
/**
tests/U4Physics.hh : Not remotely reusable so consigned to tests folder together with DsG4Scintillation
=========================================================================================================

This is intended solely for use from U4AppTest 

**/

#include <cstdlib>
#include <string>
#include <sstream>

#include "plog/Severity.h"
#include "G4VUserPhysicsList.hh"

class Local_G4Cerenkov_modified ; 
class Local_DsG4Scintillation ; 

#ifdef DEBUG_TAG
class ShimG4OpAbsorption ;
class ShimG4OpRayleigh ;
#else
class G4OpAbsorption ;
class G4OpRayleigh ;
#endif

class G4FastSimulationManagerProcess ; 

#ifdef WITH_CUSTOM4
class C4OpBoundaryProcess ; 
#elif WITH_PMTSIM
class CustomG4OpBoundaryProcess ; 
#else
class InstrumentedG4OpBoundaryProcess ; 
#endif

#include "U4_API_EXPORT.hh"

struct U4_API U4Physics : public G4VUserPhysicsList
{
    static const plog::Severity LEVEL ; 
    static int EInt(const char* key, const char* fallback="0"); 

    Local_G4Cerenkov_modified*  fCerenkov ; 
    Local_DsG4Scintillation*    fScintillation ; 

#ifdef DEBUG_TAG
    ShimG4OpAbsorption*   fAbsorption ;
    ShimG4OpRayleigh*     fRayleigh ;
#else
    G4OpAbsorption*       fAbsorption ;
    G4OpRayleigh*         fRayleigh ;
#endif

#ifdef WITH_CUSTOM4
    C4OpBoundaryProcess*  fBoundary ;
#elif WITH_PMTSIM
    CustomG4OpBoundaryProcess*  fBoundary ;
#else
    InstrumentedG4OpBoundaryProcess*  fBoundary ;
#endif

    G4FastSimulationManagerProcess*   fFastSim ;  

    static std::string Desc(); 
    U4Physics(); 

    void ConstructParticle();
    void ConstructProcess();
    void ConstructEM();
    void ConstructOp();
};



