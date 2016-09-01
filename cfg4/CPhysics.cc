#include "G4RunManager.hh"


#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif


/*


-------- EEEE ------- G4Exception-START -------- EEEE -------

*** ExceptionHandler is not defined ***
*** G4Exception : Run0041
      issued by : G4UserRunAction::G4UserRunAction()
 You are instantiating G4UserRunAction BEFORE your G4VUserPhysicsList is
instantiated and assigned to G4RunManager.
 Such an instantiation is prohibited by Geant4 version 8.0. To fix this problem,
please make sure that your main() instantiates G4VUserPhysicsList AND
set it to G4RunManager before instantiating other user action classes
such as G4UserRunAction.
*** Fatal Exception ***
-------- EEEE -------- G4Exception-END --------- EEEE -------

*/


#include "CPhysics.hh"

CPhysics::CPhysics(OpticksHub* hub) 
    :
    m_hub(hub),
    m_runManager(new G4RunManager),
#ifdef OLDPHYS
    m_physics(new PhysicsList())
#else
    m_physics(new OpNovicePhysicsList())
#endif
{
    init();
}

void CPhysics::init()
{
    m_runManager->SetUserInitialization(m_physics);
}

G4RunManager* CPhysics::getRunManager()
{
   return m_runManager ; 
}


void CPhysics::setProcessVerbosity(int verbosity)
{
   // NB processes are instanciated only after PhysicsList Construct that happens at runInitialization 
   // so this needs to be called after then
#ifdef OLDPHYS
#else
    m_physics->setProcessVerbosity(verbosity); 
#endif
}






