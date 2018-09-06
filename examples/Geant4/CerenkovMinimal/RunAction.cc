
#include <cassert>

#ifdef WITH_OPTICKS
#include "G4TransportationManager.hh"
#include "G4Opticks.hh"
#endif

#include "RunAction.hh"
#include "PLOG.hh"

RunAction::RunAction(Ctx* ctx_) 
    :   
    G4UserRunAction(),
    ctx(ctx_)
{
}
void RunAction::BeginOfRunAction(const G4Run*)
{
    LOG(info) << "." ;
#ifdef WITH_OPTICKS
    G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ; 
    assert( world ) ; 
    bool standardize_geant4_materials = true ;   // required for alignment 
    G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );    
#endif
}
void RunAction::EndOfRunAction(const G4Run*)
{
    LOG(info) << "." ;
#ifdef WITH_OPTICKS
    LOG(info) << G4Opticks::GetOpticks()->desc(); 
    G4Opticks::Finalize();
#endif
}

