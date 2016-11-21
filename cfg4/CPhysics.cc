#include "G4RunManager.hh"


#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "OpticksHub.hh"
#include "CG4.hh"
#include "CPhysics.hh"

CPhysics::CPhysics(CG4* g4) 
    :
    m_g4(g4),
    m_hub(g4->getHub()),
    m_ok(g4->getOpticks()),
    m_runManager(new G4RunManager),
#ifdef OLDPHYS
    m_physics(new PhysicsList())
#else
    m_physics(new OpNovicePhysicsList(m_g4))
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






