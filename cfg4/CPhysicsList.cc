#include <sstream>

#include "G4Version.hh"
#include "CPhysicsList.hh"
#include "G4ProcessManager.hh"

#include "CG4.hh"

#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"

#include "DsG4OpBoundaryProcess.h"
#include "Opticks.hh"

#include "PLOG.hh"


const CPhysicsList* CPhysicsList::INSTANCE = NULL ; 


CPhysicsList::CPhysicsList(CG4* g4) 
    :   
    G4VUserPhysicsList(),
    m_g4(g4),
    m_ok(g4->getOpticks()),
    m_emVerbosity(0),
    m_cerenkov(NULL),
    m_cerenkovProcess(NULL),
    m_scintillationProcess(NULL),
    m_boundaryProcess(NULL),
    m_absorptionProcess(NULL),
    m_rayleighProcess(NULL)
{
    INSTANCE = this ; 
}

CPhysicsList::~CPhysicsList() 
{
}



#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"


void CPhysicsList::ConstructParticle()
{
  // In this method, static member functions should be called
  // for all particles which you want to use.
  // This ensures that objects of these particle types will be
  // created in the program.

    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle(); 
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();


/*
  G4BosonConstructor bConstructor;
  bConstructor.ConstructParticle();

  G4LeptonConstructor lConstructor;
  lConstructor.ConstructParticle();

  G4MesonConstructor mConstructor;
  mConstructor.ConstructParticle();

  G4BaryonConstructor rConstructor;
  rConstructor.ConstructParticle();

  G4IonConstructor iConstructor;
  iConstructor.ConstructParticle();

*/

    initParticles(); 
}



// 1042 is pure guess at which version this became necessary 
#if ( G4VERSION_NUMBER >= 1042 )
void CPhysicsList::initParticles()
{
  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
      G4ParticleDefinition* particle = particleIterator->value();
      m_particles.push_back(particle); 
  }
} 
#else
void CPhysicsList::initParticles()
{
  theParticleIterator->reset();
  while( (*theParticleIterator)() )
  {
      G4ParticleDefinition* particle = theParticleIterator->value();
      m_particles.push_back(particle); 
  }
}
#endif


void CPhysicsList::ConstructProcess()
{ 
    AddTransportation();
    constructDecay();
    constructEM();
    constructOp();
}

#include "G4Decay.hh"

void CPhysicsList::constructDecay()
{
    G4Decay* theDecayProcess = new G4Decay();
    for(VP::iterator it=m_particles.begin() ; it != m_particles.end() ; it++ )
    {
        G4ParticleDefinition* particle = *it ; 
        G4ProcessManager* pmanager = particle->GetProcessManager();
        if (theDecayProcess->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(theDecayProcess);
            pmanager->SetProcessOrdering(theDecayProcess, idxPostStep);
            pmanager->SetProcessOrdering(theDecayProcess, idxAtRest);
        }
    }
}

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4eMultipleScattering.hh"
#include "G4MuMultipleScattering.hh"
#include "G4hMultipleScattering.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"

#include "G4hIonisation.hh"


void CPhysicsList::setupEmVerbosity(unsigned verbosity)
{
   // these are used in Em process constructors, so do this prior to process creation
    G4EmParameters* empar = G4EmParameters::Instance() ;
    empar->SetVerbose(verbosity); 
    empar->SetWorkerVerbose(verbosity); 
}

void CPhysicsList::constructEM()
{
    setupEmVerbosity(m_emVerbosity);
    for(VP::iterator it=m_particles.begin() ; it != m_particles.end() ; it++ ) constructEM(*it) ; 
}
 
void CPhysicsList::constructEM( G4ParticleDefinition* particle )
{
    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();

    if (particleName == "gamma") 
    {
        pmanager->AddDiscreteProcess(new G4GammaConversion());
        pmanager->AddDiscreteProcess(new G4ComptonScattering());
        pmanager->AddDiscreteProcess(new G4PhotoElectricEffect());
    } 
    else if (particleName == "e-") 
    {
        pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1);
        pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2);
        pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3);
    } 
    else if (particleName == "e+") 
    {
        pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1);
        pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2);
        pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3);
        pmanager->AddProcess(new G4eplusAnnihilation(),  0,-1, 4);
    } 
    else if( particleName == "mu+" || particleName == "mu-" ) 
    {
        pmanager->AddProcess(new G4MuMultipleScattering(),-1, 1, 1);
        pmanager->AddProcess(new G4MuIonisation(),      -1, 2, 2);
        pmanager->AddProcess(new G4MuBremsstrahlung(),  -1, 3, 3);
        pmanager->AddProcess(new G4MuPairProduction(),  -1, 4, 4);
    } 
    else 
    {
        if (
              (particle->GetPDGCharge() != 0.0) &&
              (particle->GetParticleName() != "chargedgeantino") &&
              !particle->IsShortLived()
           ) 
        {
            // all others charged particles except geantino
            pmanager->AddProcess(new G4hMultipleScattering(),-1,1,1);
            pmanager->AddProcess(new G4hIonisation(),       -1,2,2);
        }
   }
}




#include "CCerenkov.hh"
#include "G4OpBoundaryProcess.hh"


void CPhysicsList::constructOp()
{
    m_cerenkov = new CCerenkov(m_g4); 
    m_cerenkovProcess = m_cerenkov->getProcess() ; 

    m_absorptionProcess = new G4OpAbsorption();
    m_rayleighProcess = new G4OpRayleigh();


#ifdef USE_CUSTOM_BOUNDARY
    m_boundaryProcess = new DsG4OpBoundaryProcess(m_g4) ;
#else
    m_boundaryProcess = new G4OpBoundaryProcess() ;
#endif

    LOG(info) << description() ; 


    for(VP::iterator it=m_particles.begin() ; it != m_particles.end() ; it++ ) constructOp(*it) ; 
}

void CPhysicsList::constructOp( G4ParticleDefinition* particle )
{
    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();

    if ( m_cerenkovProcess && m_cerenkovProcess->IsApplicable(*particle))
    {
        pmanager->AddProcess(m_cerenkovProcess);
        pmanager->SetProcessOrdering(m_cerenkovProcess,idxPostStep);
    }

    if ( m_scintillationProcess && m_scintillationProcess->IsApplicable(*particle))
    {
        pmanager->AddProcess(m_scintillationProcess);
        pmanager->SetProcessOrderingToLast(m_scintillationProcess, idxAtRest);
        pmanager->SetProcessOrderingToLast(m_scintillationProcess, idxPostStep);
    }

    if (particleName == "opticalphoton")
    {
        if(m_absorptionProcess) 
        pmanager->AddDiscreteProcess(m_absorptionProcess);

        if(m_rayleighProcess)
        pmanager->AddDiscreteProcess(m_rayleighProcess);

        //pmanager->AddDiscreteProcess(fMieHGScatteringProcess);

        assert(m_boundaryProcess); 
        pmanager->AddDiscreteProcess(m_boundaryProcess);
    }
}


std::string CPhysicsList::description() const
{
    std::stringstream ss ; 
    ss << "CPhysicsList ( "  ; 
#ifdef USE_CUSTOM_BOUNDARY
    ss << "USE_CUSTOM_BOUNDARY DsG4OpBoundaryProcess " ;  
#else
    ss << "G4OpBoundaryProcess " ; 
#endif
    if(m_absorptionProcess) ss << "G4OpAbsorption " ; 
    if(m_rayleighProcess) ss << "G4OpRayleigh " ; 
    ss << ")" ; 
    return ss.str(); 
}


void CPhysicsList::setProcessVerbosity(int verbosity)
{
    for(VP::iterator it=m_particles.begin() ; it != m_particles.end() ; it++ ) setProcessVerbosity(*it, verbosity) ; 
}

void CPhysicsList::setProcessVerbosity(G4ParticleDefinition* particle, int verbosity)
{
    G4String particleName = particle->GetParticleName();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    if(!pmanager) return ; 

    unsigned int npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
    G4ProcessVector* procs = pmanager->GetProcessList();
    for(unsigned int i=0 ; i < npro ; i++)
    {
        G4VProcess* proc = (*procs)[i] ; 
        G4String processName = proc->GetProcessName() ;
        G4int prior = proc->GetVerboseLevel();
        proc->SetVerboseLevel(verbosity);
        G4int curr = proc->GetVerboseLevel() ;
        assert(curr == verbosity);

        if(curr != prior)
            LOG(debug) << "CPhysicsList::setProcessVerbosity " << particleName << ":" << processName << " from " << prior << " to " << proc->GetVerboseLevel() ;

    }
}


