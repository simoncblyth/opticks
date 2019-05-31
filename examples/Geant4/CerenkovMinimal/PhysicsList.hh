#pragma once

#include "G4VUserPhysicsList.hh"

class G4Scintillation ;
class G4OpBoundaryProcess ;
class G4OpAbsorption ;
class G4OpRayleigh ;

template <typename T>
struct PhysicsList : public G4VUserPhysicsList
{
    PhysicsList();

    virtual void ConstructParticle();
    virtual void ConstructProcess();

    void ConstructEM();
    void ConstructOp();

    G4int                fMaxNumPhotonStep ; 
    G4int                fVerboseLevel ;  
    T*                   fCerenkovProcess ; 
    G4Scintillation*     fScintillationProcess ; 
    G4OpAbsorption*      fAbsorptionProcess;
    G4OpRayleigh*        fRayleighProcess;
    G4OpBoundaryProcess* fBoundaryProcess ; 
};





