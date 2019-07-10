#include "G4RunManager.hh"


#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "CG4.hh"
#include "CPhysics.hh"
#include "CPhysicsList.hh"

int CPhysics::preinit()
{
    OK_PROFILE("_CPhysics::CPhysics"); 
    return 0 ; 
}

CPhysics::CPhysics(CG4* g4) 
    :
    m_g4(g4),
    m_hub(g4->getHub()),
    m_ok(g4->getOpticks()),
    m_preinit(preinit()),
    m_runManager(new G4RunManager),
#ifdef OLDPHYS
    m_physicslist(new PhysicsList())
#else
    m_physicslist(new CPhysicsList(m_g4))
    //m_physicslist(new OpNovicePhysicsList(m_g4))
#endif
{
    init();
}

void CPhysics::init()
{
    OK_PROFILE("CPhysics::CPhysics"); 
    m_runManager->SetNumberOfEventsToBeStored(0); 
    m_runManager->SetUserInitialization(m_physicslist);
}

G4RunManager* CPhysics::getRunManager() const 
{
   return m_runManager ; 
}

void CPhysics::setProcessVerbosity(int verbosity)
{
   // NB processes are instanciated only after PhysicsList Construct that happens at runInitialization 
   // so this needs to be called after then
#ifdef OLDPHYS
#else
    m_physicslist->setProcessVerbosity(verbosity); 
#endif
}



