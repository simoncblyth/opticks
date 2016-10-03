#include "G4RunManager.hh"


#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "OpticksHub.hh"
#include "CPhysics.hh"

CPhysics::CPhysics(OpticksHub* hub) 
    :
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_runManager(new G4RunManager),
#ifdef OLDPHYS
    m_physics(new PhysicsList())
#else
    m_physics(new OpNovicePhysicsList(m_ok))
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






