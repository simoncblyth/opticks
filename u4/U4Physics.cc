/**
U4Physics.cc
==============

Boundary class changes need to match in all the below::

    U4OpBoundaryProcess.h    
    U4Physics.cc 
    U4Recorder.cc
    U4StepPoint.cc

**/


#include <iomanip>
#include "ssys.h"
#include "U4Physics.hh"
#include "U4OpBoundaryProcess.h"
#include "G4ProcessManager.hh"
#include "G4FastSimulationManagerProcess.hh"


#if defined(WITH_CUSTOM4) && defined(WITH_PMTSIM)
#include "G4OpBoundaryProcess.hh"
#include "C4OpBoundaryProcess.hh"
#include "PMTSimParamSvc/PMTAccessor.h"
#elif defined(WITH_CUSTOM4) && !defined(WITH_PMTSIM)
#include "G4OpBoundaryProcess.hh"
#include "C4OpBoundaryProcess.hh"
#include "SPMTAccessor.h"
#elif defined(WITH_INSTRUMENTED_DEBUG)
#include "InstrumentedG4OpBoundaryProcess.hh"
#else
#include "G4OpBoundaryProcess.hh"
#endif


#include "SLOG.hh"
const plog::Severity U4Physics::LEVEL = SLOG::EnvLevel("U4Physics", "DEBUG") ; 

U4Physics::U4Physics()
    :
    fCerenkov(nullptr),
    fScintillation(nullptr),
    fAbsorption(nullptr),
    fRayleigh(nullptr),
    fBoundary(nullptr),
    fFastSim(nullptr)
{
    Cerenkov_DISABLE = EInt(_Cerenkov_DISABLE, "0") ; 
    Scintillation_DISABLE = EInt(_Scintillation_DISABLE, "0" );
    OpAbsorption_DISABLE = EInt(_OpAbsorption_DISABLE, "0") ; 
    OpRayleigh_DISABLE = EInt(_OpRayleigh_DISABLE, "0") ; 
    OpBoundaryProcess_DISABLE = EInt(_OpBoundaryProcess_DISABLE, "0") ; 
    OpBoundaryProcess_LASTPOST = EInt(_OpBoundaryProcess_LASTPOST, "0") ; 
    FastSim_ENABLE = EInt(_FastSim_ENABLE, "0") ; 
}




#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"


void U4Physics::ConstructParticle()
{
    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle();
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();
}

void U4Physics::ConstructProcess()
{
    AddTransportation();
    ConstructEM();
    ConstructOp();
}

// from OpNovicePhysicsList::ConstructEM

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

void U4Physics::ConstructEM()
{
    G4int em_verbosity = 0 ; 
    G4EmParameters* empar = G4EmParameters::Instance() ;
    empar->SetVerbose(em_verbosity); 
    empar->SetWorkerVerbose(em_verbosity); 

  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
    G4ParticleDefinition* particle = particleIterator->value();

    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();

    if (particleName == "gamma") {
    // gamma
      // Construct processes for gamma
      pmanager->AddDiscreteProcess(new G4GammaConversion());
      pmanager->AddDiscreteProcess(new G4ComptonScattering());
      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect());

    } else if (particleName == "e-") {
    //electron
      // Construct processes for electron
      pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1); 
      pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2); 
      pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3); 

    } else if (particleName == "e+") {
    //positron
      // Construct processes for positron
      pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1); 
      pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2); 
      pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3); 
      pmanager->AddProcess(new G4eplusAnnihilation(),  0,-1, 4); 

    } else if( particleName == "mu+" ||
               particleName == "mu-"    ) { 
    //muon
     // Construct processes for muon
     pmanager->AddProcess(new G4MuMultipleScattering(),-1, 1, 1); 
     pmanager->AddProcess(new G4MuIonisation(),      -1, 2, 2); 
     pmanager->AddProcess(new G4MuBremsstrahlung(),  -1, 3, 3); 
     pmanager->AddProcess(new G4MuPairProduction(),  -1, 4, 4); 

    } else {
      if ((particle->GetPDGCharge() != 0.0) &&
          (particle->GetParticleName() != "chargedgeantino") &&
          !particle->IsShortLived()) {
       // all others charged particles except geantino
       pmanager->AddProcess(new G4hMultipleScattering(),-1,1,1);
       pmanager->AddProcess(new G4hIonisation(),       -1,2,2);
     }   
    }   
  }
}

#include "Local_G4Cerenkov_modified.hh"
#include "Local_DsG4Scintillation.hh"

#include "ShimG4OpAbsorption.hh"
#include "ShimG4OpRayleigh.hh"


std::string U4Physics::desc() const
{
    std::stringstream ss ; 
    ss
        << "U4Physics::desc" << "\n"      
        << std::setw(60) << _Cerenkov_DISABLE           << " : " << Cerenkov_DISABLE << "\n"
        << std::setw(60) << _Scintillation_DISABLE      << " : " << Scintillation_DISABLE << "\n"
        << std::setw(60) << _OpAbsorption_DISABLE       << " : " << OpAbsorption_DISABLE << "\n" 
        << std::setw(60) << _OpRayleigh_DISABLE         << " : " << OpRayleigh_DISABLE << "\n"
        << std::setw(60) << _OpBoundaryProcess_DISABLE  << " : " << OpBoundaryProcess_DISABLE << "\n"
        << std::setw(60) << _OpBoundaryProcess_LASTPOST << " : " << OpBoundaryProcess_LASTPOST << "\n"
        << std::setw(60) << _FastSim_ENABLE             << " : " << FastSim_ENABLE << "\n"
        ;
    std::string str = ss.str();
    return str ;
}


std::string U4Physics::Desc()  // static 
{
    std::stringstream ss ; 
#ifdef DEBUG_TAG
    ss << ( ShimG4OpAbsorption::FLOAT ? "ShimG4OpAbsorption_FLOAT" : "ShimG4OpAbsorption_ORIGINAL" ) ; 
    ss << "_" ; 
    ss << ( ShimG4OpRayleigh::FLOAT ? "ShimG4OpRayleigh_FLOAT" : "ShimG4OpRayleigh_ORIGINAL" ) ; 
#endif
    std::string str = ss.str();
    return str ;
}




std::string U4Physics::Switches()  // static 
{
    std::stringstream ss ; 
    ss << "U4Physics::Switches" << std::endl ; 
#if defined(WITH_CUSTOM4)
    ss << "WITH_CUSTOM4" << std::endl ; 
#else
    ss << "NOT:WITH_CUSTOM4" << std::endl ; 
#endif
#if defined(WITH_PMTSIM)
    ss << "WITH_PMTSIM" << std::endl ; 
#else
    ss << "NOT:WITH_PMTSIM" << std::endl ; 
#endif
#if defined(WITH_CUSTOM4) && defined(WITH_PMTSIM)
    ss << "WITH_CUSTOM4_AND_WITH_PMTSIM" << std::endl ; 
#else
    ss << "NOT:WITH_CUSTOM4_AND_WITH_PMTSIM" << std::endl ; 
#endif

#if defined(WITH_CUSTOM4) && !defined(WITH_PMTSIM)
    ss << "WITH_CUSTOM4_AND_NOT_WITH_PMTSIM" << std::endl ; 
#else
    ss << "NOT:WITH_CUSTOM4_AND_NOT_WITH_PMTSIM" << std::endl ; 
#endif

#if defined(DEBUG_TAG)
    ss << "DEBUG_TAG" << std::endl ; 
#else
    ss << "NOT:DEBUG_TAG" << std::endl ; 
#endif
    std::string str = ss.str();
    return str ;
}



int U4Physics::EInt(const char* key, const char* fallback)  // static 
{
    const char* val_ = getenv(key) ;
    int val =  std::atoi(val_ ? val_ : fallback) ;
    return val ; 
}


/**
U4Physics::ConstructOp
-----------------------

Scintillation needs to come after absorption for reemission
to sometimes happen for fStopAndKill 

But suspect coming after boundary may be causing 
the need for UseGivenVelocity_KLUDGE to get velocity and times correct see::

    ~/o/notes/issues/Geant4_UseGivenVelocity_after_refraction_is_there_a_better_way_than_the_kludge_fix.rst 


**/


void U4Physics::ConstructOp()
{
    LOG(info) << desc() ;  

    if(Cerenkov_DISABLE == 0)
    {
        fCerenkov = new Local_G4Cerenkov_modified ;
        fCerenkov->SetMaxNumPhotonsPerStep(10000);
        fCerenkov->SetMaxBetaChangePerStep(10.0);
        fCerenkov->SetTrackSecondariesFirst(true);   
        fCerenkov->SetVerboseLevel(EInt("Local_G4Cerenkov_modified_verboseLevel", "0"));
    }

    if(Scintillation_DISABLE == 0)
    {
        fScintillation = new Local_DsG4Scintillation(EInt("Local_DsG4Scintillation_opticksMode","0")) ; 
        fScintillation->SetTrackSecondariesFirst(true);
    }

    if(FastSim_ENABLE == 1 )
    {
        fFastSim  = new G4FastSimulationManagerProcess("fast_sim_man");
    }

    if(OpAbsorption_DISABLE == 0)
    {
#ifdef DEBUG_TAG
        fAbsorption = new ShimG4OpAbsorption();
#else
        fAbsorption = new G4OpAbsorption();
#endif
    }

    if(OpRayleigh_DISABLE == 0)
    {
#ifdef DEBUG_TAG
        fRayleigh = new ShimG4OpRayleigh();
#else
        fRayleigh = new G4OpRayleigh();
#endif
    }

    if(OpBoundaryProcess_DISABLE == 0)
    {
        fBoundary = CreateBoundaryProcess(); 
        LOG(info) << " fBoundary " << fBoundary ; 
    }



  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        if ( fCerenkov && fCerenkov->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(fCerenkov);
            pmanager->SetProcessOrdering(fCerenkov,idxPostStep);
        }

        if ( fScintillation && fScintillation->IsApplicable(*particle) && particleName != "opticalphoton") 
        {
            pmanager->AddProcess(fScintillation);
            pmanager->SetProcessOrderingToLast(fScintillation, idxAtRest);
            pmanager->SetProcessOrderingToLast(fScintillation, idxPostStep);
        }

        if (particleName == "opticalphoton") 
        {
            ConstructOp_opticalphoton(pmanager, particleName); 
        }
    }
}


/**
U4Physics::ConstructOp_opticalphoton
------------------------------------

TO AVOID THE UseGivenVelocity KLUDGE scintillation process needs to be after absorption
BUT UNFORTUNATELY PUTTING Scintillation AFTER Absorption prevents REEMISSION from happening

* ~/o/notes/issues/Geant4_UseGivenVelocity_KLUDGE_may_be_avoided_by_doing_PostStepDoIt_for_boundary_after_scintillation
* ~/o/notes/issues/G4CXTest_GEOM_shakedown.rst

**/

void U4Physics::ConstructOp_opticalphoton(G4ProcessManager* pmanager, const G4String& particleName)
{
    assert( particleName == "opticalphoton" ); 

    if(fScintillation) 
    {
        pmanager->AddProcess(fScintillation); 
        pmanager->SetProcessOrderingToLast(fScintillation, idxAtRest);
        pmanager->SetProcessOrderingToLast(fScintillation, idxPostStep);
    }
    if(fAbsorption)    pmanager->AddDiscreteProcess(fAbsorption);
    if(fRayleigh)      pmanager->AddDiscreteProcess(fRayleigh);
    if(fBoundary)      pmanager->AddDiscreteProcess(fBoundary);
    if(fFastSim)       pmanager->AddDiscreteProcess(fFastSim); 
}






/**
U4Physics::CreateBoundaryProcess
---------------------------------

Looks like this needs updating now that it
is normal to use WITH_CUSTOM4 within junosw+opticks
without using WITH_PMTSIM

* NB : BOUNDARY CLASS CHANGES HERE MUST PARALLEL THOSE IN U4OpBoundaryProcess.h

  * OTHERWISE GET "UNEXPECTED BoundaryFlag ZERO "

**/

G4VProcess* U4Physics::CreateBoundaryProcess()  // static 
{
    G4VProcess* proc = nullptr ; 

#if defined(WITH_PMTSIM) && defined(WITH_CUSTOM4)
    const char* path = "$PMTSimParamData_BASE" ;  // directory with PMTSimParamData subfolder
    const PMTSimParamData* data = PMTAccessor::LoadData(path) ; 
    LOG(LEVEL) << "load path "  << path << " giving PMTSimParamData.data: " << ( data ? "YES" : "NO" ) ; 
    //LOG_IF(LEVEL, data != nullptr ) << *data ; 

    const PMTAccessor* pmt = PMTAccessor::Create(data) ; 
    const C4IPMTAccessor* ipmt = pmt ;  
    proc = new C4OpBoundaryProcess(ipmt);

    LOG(LEVEL) << "create C4OpBoundaryProcess :  WITH_CUSTOM4 WITH_PMTSIM " ; 

#elif defined(WITH_CUSTOM4)
    //const U4PMTAccessor* pmt = new U4PMTAccessor ; // DUMMY PLACEHOLDER
    const char* jpmt = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt" ; 
    const SPMTAccessor* pmt = SPMTAccessor::Load(jpmt) ; 
    const char* geom = ssys::getenvvar("GEOM", "no-GEOM") ; 
    LOG_IF(fatal, pmt == nullptr ) 
         << " FAILED TO SPMTAccessor::Load from [" << jpmt << "]" 
         << " GEOM " << ( geom ? geom : "-" )      
         ; 

    // assert(pmt) ;  // trying to get C4 to work without the PMT info, just assert when really need PMT info 
    const C4IPMTAccessor* ipmt = pmt ;  
    proc = new C4OpBoundaryProcess(ipmt);
    LOG(LEVEL) << "create C4OpBoundaryProcess :  WITH_CUSTOM4 NOT:WITH_PMTSIM " ; 

#elif defined(WITH_INSTRUMENTED_DEBUG)
    proc = new InstrumentedG4OpBoundaryProcess();
    LOG(LEVEL) << "create InstrumentedG4OpBoundaryProcess : NOT (WITH_PMTSIM and WITH_CUSTOM4) " ; 
#else
    proc = new G4OpBoundaryProcess();
    //LOG(LEVEL) << "create G4OpBoundaryProcess : NOT (WITH_PMTSIM and WITH_CUSTOM4) " ; 
#endif
    return proc ; 
}

