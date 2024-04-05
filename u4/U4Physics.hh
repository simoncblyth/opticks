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

class G4VProcess ; 
class G4FastSimulationManagerProcess ; 


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

    G4VProcess*          fBoundary ; 
    G4FastSimulationManagerProcess*   fFastSim ;  

    std::string desc() const ; 
    static std::string Desc(); 
    static std::string Switches(); 

    U4Physics(); 

    void ConstructParticle();
    void ConstructProcess();
    void ConstructEM();
    void ConstructOp();
    static G4VProcess* CreateBoundaryProcess(); 

    static constexpr const char* _Cerenkov_DISABLE = "U4Physics__ConstructOp_Cerenkov_DISABLE" ; 
    static constexpr const char* _Scintillation_DISABLE = "U4Physics__ConstructOp_Scintillation_DISABLE" ; 
    static constexpr const char* _OpAbsorption_DISABLE = "U4Physics__ConstructOp_OpAbsorption_DISABLE" ; 
    static constexpr const char* _OpRayleigh_DISABLE = "U4Physics__ConstructOp_OpRayleigh_DISABLE" ; 
    static constexpr const char* _OpBoundaryProcess_DISABLE = "U4Physics__ConstructOp_OpBoundaryProcess_DISABLE" ; 
    static constexpr const char* _OpBoundaryProcess_LASTPOST = "U4Physics__ConstructOp_OpBoundaryProcess_LASTPOST" ; 
    static constexpr const char* _FastSim_ENABLE = "U4Physics__ConstructOp_FastSim_ENABLE" ; 

    int Cerenkov_DISABLE = 0 ; 
    int Scintillation_DISABLE = 0 ; 
    int OpAbsorption_DISABLE = 0 ; 
    int OpRayleigh_DISABLE = 0 ; 
    int OpBoundaryProcess_DISABLE = 0 ; 
    int OpBoundaryProcess_LASTPOST = 0 ; 
    int FastSim_ENABLE = 0 ; 
};


