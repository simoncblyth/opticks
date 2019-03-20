
#include <cassert>

#ifdef WITH_OPTICKS
#include "G4TransportationManager.hh"
#include "G4Opticks.hh"
#endif

#include "RunAction.hh"

RunAction::RunAction(Ctx* ctx_) 
    :   
    G4UserRunAction(),
    ctx(ctx_)
{
}
void RunAction::BeginOfRunAction(const G4Run*)
{
#ifdef WITH_OPTICKS
    G4cout << "\n\n###[ RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ; 
    G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ; 
    assert( world ) ; 
    bool standardize_geant4_materials = true ;   // required for alignment 
    G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );    
    G4cout << "\n\n###] RunAction::BeginOfRunAction G4Opticks.setGeometry\n\n" << G4endl ; 
#endif
}
void RunAction::EndOfRunAction(const G4Run*)
{
#ifdef WITH_OPTICKS
    G4cout << "\n\n###[ RunAction::EndOfRunAction G4Opticks.Finalize\n\n" << G4endl ; 
    G4Opticks::Finalize();
    G4cout << "\n\n###] RunAction::EndOfRunAction G4Opticks.Finalize\n\n" << G4endl ; 
#endif
}

