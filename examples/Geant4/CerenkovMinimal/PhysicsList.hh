#pragma once

#include "G4VUserPhysicsList.hh"

class G4Cerenkov ;
class G4Scintillation ;
class G4OpBoundaryProcess ;

struct PhysicsList : public G4VUserPhysicsList
{
    PhysicsList();

    virtual void ConstructParticle();
    virtual void ConstructProcess();

    G4int                fMaxNumPhotonStep ; 
    G4int                fVerboseLevel ;  
    G4Cerenkov*          fCerenkovProcess ; 
    G4Scintillation*     fScintillationProcess ; 
    G4OpBoundaryProcess* fBoundaryProcess ; 
};





