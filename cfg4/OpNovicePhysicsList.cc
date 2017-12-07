#include "CFG4_BODY.hh"

#include "CFG4_PUSH.hh"

#include "globals.hh"
#include "OpNovicePhysicsList.hh"
#include "OpNovicePhysicsListMessenger.hh"
#include "DebugG4Transportation.hh"
#include "G4RunManagerKernel.hh"


#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4ProcessManager.hh"
#include "G4FastSimulationManagerProcess.hh"

#include "G4OpAbsorption.hh"
#include "G4OpMieHG.hh"


#ifdef USE_POWER_THIRD_RAYLEIGH
#include "DsG4OpRayleigh.h"
#else
#include "OpRayleigh.hh"
#endif


#include "CBoundaryProcess.hh"

#ifdef USE_CUSTOM_CERENKOV
#include "DsG4Cerenkov.h"
#else
#include "G4Cerenkov.hh"
#endif

#ifdef USE_CUSTOM_SCINTILLATION
#include "DsG4Scintillation.h"
#else
#include "G4Scintillation.hh"
#endif


#include "G4LossTableManager.hh"
#include "G4EmSaturation.hh"

#include "CFG4_POP.hh"

#include "Opticks.hh"
// cfg4-
#include "CG4.hh"
#include "PLOG.hh"
#include "CMPT.hh"
#include "OpRayleigh.hh"
#include "Scintillation.hh"
#include "Cerenkov.hh"


using CLHEP::g ; 
using CLHEP::cm2 ; 
using CLHEP::MeV ; 
using CLHEP::ns ; 




G4ThreadLocal G4int OpNovicePhysicsList::fVerboseLevel = 1;
G4ThreadLocal G4int OpNovicePhysicsList::fMaxNumPhotonStep = 20;


#ifdef USE_CUSTOM_CERENKOV
#  ifndef USE_CUSTOM_WITHGENSTEP_CERENKOV
G4ThreadLocal DsG4Cerenkov* OpNovicePhysicsList::fCerenkovProcess = 0;
#  else
G4ThreadLocal Cerenkov* OpNovicePhysicsList::fCerenkovProcess = 0;
#  endif
#else
G4ThreadLocal G4Cerenkov* OpNovicePhysicsList::fCerenkovProcess = 0;
#endif

#ifdef USE_CUSTOM_SCINTILLATION
#  ifndef USE_CUSTOM_WITHGENSTEP_SCINTILLATION
G4ThreadLocal DsG4Scintillation* OpNovicePhysicsList::fScintillationProcess = 0;
#  else
G4ThreadLocal Scintillation* OpNovicePhysicsList::fScintillationProcess = 0;
#  endif
#else
G4ThreadLocal G4Scintillation* OpNovicePhysicsList::fScintillationProcess = 0;
#endif

#ifdef USE_CUSTOM_BOUNDARY
G4ThreadLocal DsG4OpBoundaryProcess* OpNovicePhysicsList::fBoundaryProcess = 0;
#else
G4ThreadLocal G4OpBoundaryProcess* OpNovicePhysicsList::fBoundaryProcess = 0;
#endif
 

G4ThreadLocal G4OpAbsorption* OpNovicePhysicsList::fAbsorptionProcess = 0;

#ifdef USE_POWER_THIRD_RAYLEIGH
G4ThreadLocal DsG4OpRayleigh* OpNovicePhysicsList::fRayleighScatteringProcess = 0;
#else
G4ThreadLocal OpRayleigh* OpNovicePhysicsList::fRayleighScatteringProcess = 0;
#endif

#ifdef USE_DEBUG_TRANSPORTATION
G4ThreadLocal DebugG4Transportation* OpNovicePhysicsList::fTransportationProcess = 0;
#else
G4ThreadLocal G4Transportation*      OpNovicePhysicsList::fTransportationProcess = 0;
#endif


G4ThreadLocal G4OpMieHG* OpNovicePhysicsList::fMieHGScatteringProcess = 0;



OpNovicePhysicsList::OpNovicePhysicsList(CG4* g4) 
    : 
    G4VUserPhysicsList(),
    m_g4(g4),
    m_ok(g4->getOpticks()),
    // below defaults from DsPhysConsOptical
    m_doReemission(true),               // "ScintDoReemission"        "Do reemission in scintilator."
    m_doScintAndCeren(true),            // "ScintDoScintAndCeren"     "Do both scintillation and Cerenkov in scintilator."
    m_useFastMu300nsTrick(false),       // "UseFastMu300nsTrick"      "Use Fast muon simulation?"
    m_useCerenkov(true),                // "UseCerenkov"              "Use the Cerenkov process?"
    m_useScintillation(true),           // "UseScintillation"         "Use the Scintillation process?"
    m_useRayleigh(true),                // "UseRayleigh"              "Use the Rayleigh scattering process?"
    m_useAbsorption(true),              // "UseAbsorption"            "Use light absorption process?"
    m_applyWaterQe(true),               // "ApplyWaterQe"             
                                        // "Apply QE for water cerenkov process when OP is created? If it is true the CerenPhotonScaleWeight will be disabled in water, but it still works for AD and others "
    m_cerenPhotonScaleWeight(3.125),    // "CerenPhotonScaleWeight"    "Scale down number of produced Cerenkov photons by this much."
    m_cerenMaxPhotonPerStep(300),       // "CerenMaxPhotonsPerStep"   "Limit step to at most this many (unscaled) Cerenkov photons."
    m_scintPhotonScaleWeight(3.125),    // "ScintPhotonScaleWeight"    "Scale down number of produced scintillation photons by this much."
    m_ScintillationYieldFactor(1.0),    // "ScintillationYieldFactor" "Scale the number of scintillation photons per MeV by this much."
    m_birksConstant1(6.5e-3*g/cm2/MeV), // "BirksConstant1"           "Birks constant C1"
    m_birksConstant2(3.0e-6*(g/cm2/MeV)*(g/cm2/MeV)),  
                                          // "BirksConstant2"            "Birks constant C2" 
    m_gammaSlowerTime(149*ns),          // "GammaSlowerTime"           "Gamma Slower time constant"
    m_gammaSlowerRatio(0.338),          // "GammaSlowerRatio"          "Gamma Slower time ratio"
    m_neutronSlowerTime(220*ns),        // "NeutronSlowerTime"         "Neutron Slower time constant"
    m_neutronSlowerRatio(0.34),         // "NeutronSlowerRatio"        "Neutron Slower time ratio"
    m_alphaSlowerTime(220*ns),          // "AlphaSlowerTime"           "Alpha Slower time constant"
    m_alphaSlowerRatio(0.35)            // "AlphaSlowerRatio"          "Alpha Slower time ratio"
{
    fMessenger = new OpNovicePhysicsListMessenger(this) ;
}


void OpNovicePhysicsList::dumpParam(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info)<<"Photons prescaling is "<<( m_cerenPhotonScaleWeight>1.?"on":"off" )
             <<" for Cerenkov. Preliminary applied efficiency is "
             <<1./m_cerenPhotonScaleWeight<<" (weight="<<m_cerenPhotonScaleWeight<<")" ;
    LOG(info)<<"Photons prescaling is "<<( m_scintPhotonScaleWeight>1.?"on":"off" )
             <<" for Scintillation. Preliminary applied efficiency is "
             <<1./m_scintPhotonScaleWeight<<" (weight="<<m_scintPhotonScaleWeight<<")";
    LOG(info)<<"WaterQE is turned "<<(m_applyWaterQe?"on":"off")<<" for Cerenkov.";
}


void OpNovicePhysicsList::ConstructProcess()
{
  setupEmVerbosity(0); 

  //AddTransportation();
  addTransportation();


  ConstructDecay();
  ConstructEM();

  ConstructOpDYB();

  dump("OpNovicePhysicsList::ConstructProcess"); 
}



void OpNovicePhysicsList::addTransportation()
{
   // adpated from 
   // /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc

  G4int verboseLevelTransport = 0;
  G4int nParaWorld = G4RunManagerKernel::GetRunManagerKernel()->GetNumberOfParallelWorld();
  assert(nParaWorld == 0); 

#ifdef USE_DEBUG_TRANSPORTATION  
  fTransportationProcess = new DebugG4Transportation(m_g4, verboseLevelTransport);
#else
  fTransportationProcess = new G4Transportation(verboseLevelTransport);
#endif
 
  // loop over all particles in G4ParticleTable
  theParticleIterator->reset();
  while( (*theParticleIterator)() )
  {
      G4ParticleDefinition* particle = theParticleIterator->value();
      G4ProcessManager* pmanager = particle->GetProcessManager();
      // Add transportation process for all particles 
      assert( pmanager );

      // add transportation with ordering = ( -1, "first", "first" )
      pmanager ->AddProcess(fTransportationProcess);
      pmanager ->SetProcessOrderingToFirst(fTransportationProcess, idxAlongStep);
      pmanager ->SetProcessOrderingToFirst(fTransportationProcess, idxPostStep);
  }
}


void OpNovicePhysicsList::ConstructOpDYB()
{
#ifdef USE_CUSTOM_CERENKOV
#  ifndef USE_CUSTOM_WITHGENSTEP_CERENKOV    
    LOG(info)  << "Using customized DsG4Cerenkov." ;
    DsG4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) 
    {
        cerenkov = new DsG4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetApplyPreQE(m_cerenPhotonScaleWeight>1.);
        cerenkov->SetPreQE(1./m_cerenPhotonScaleWeight);
        cerenkov->SetApplyWaterQe(m_applyWaterQe);
        cerenkov->SetTrackSecondariesFirst(true);
    }
#  else
    LOG(info)  << "Using customized Cerenkov." ;
    Cerenkov* cerenkov = 0;
    if (m_useCerenkov) 
    {
        cerenkov = new Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        // cerenkov->SetApplyPreQE(m_cerenPhotonScaleWeight>1.);
        // cerenkov->SetPreQE(1./m_cerenPhotonScaleWeight);
        // cerenkov->SetApplyWaterQe(m_applyWaterQe);
        cerenkov->SetTrackSecondariesFirst(true);
    }

#  endif
#else
    LOG(info) << "Using standard G4Cerenkov." ;
    G4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) 
    {
        cerenkov = new G4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetTrackSecondariesFirst(true);
    }
#endif
    fCerenkovProcess = cerenkov ; 

#ifdef USE_CUSTOM_SCINTILLATION
#  ifndef USE_CUSTOM_WITHGENSTEP_SCINTILLATION
    DsG4Scintillation* scint = 0;
    LOG(info) << "Using customized DsG4Scintillation." ;
    scint = new DsG4Scintillation();
    scint->SetBirksConstant1(m_birksConstant1);
    scint->SetBirksConstant2(m_birksConstant2);
    scint->SetGammaSlowerTimeConstant(m_gammaSlowerTime);
    scint->SetGammaSlowerRatio(m_gammaSlowerRatio);
    scint->SetNeutronSlowerTimeConstant(m_neutronSlowerTime);
    scint->SetNeutronSlowerRatio(m_neutronSlowerRatio);
    scint->SetAlphaSlowerTimeConstant(m_alphaSlowerTime);
    scint->SetAlphaSlowerRatio(m_alphaSlowerRatio);
    scint->SetDoReemission(m_doReemission);
    scint->SetDoBothProcess(m_doScintAndCeren);
    scint->SetApplyPreQE(m_scintPhotonScaleWeight>1.);
    scint->SetPreQE(1./m_scintPhotonScaleWeight);
    scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); //1.);
    scint->SetUseFastMu300nsTrick(m_useFastMu300nsTrick);
    scint->SetTrackSecondariesFirst(true);
    if (!m_useScintillation) scint->SetNoOp();
#  else
    Scintillation* scint = 0;
    LOG(info) << "Using customized Scintillation." ;
    scint = new Scintillation();
    // scint->SetBirksConstant1(m_birksConstant1);
    // scint->SetBirksConstant2(m_birksConstant2);
    // scint->SetGammaSlowerTimeConstant(m_gammaSlowerTime);
    // scint->SetGammaSlowerRatio(m_gammaSlowerRatio);
    // scint->SetNeutronSlowerTimeConstant(m_neutronSlowerTime);
    // scint->SetNeutronSlowerRatio(m_neutronSlowerRatio);
    // scint->SetAlphaSlowerTimeConstant(m_alphaSlowerTime);
    // scint->SetAlphaSlowerRatio(m_alphaSlowerRatio);
    // scint->SetDoReemission(m_doReemission);
    // scint->SetDoBothProcess(m_doScintAndCeren);
    // scint->SetApplyPreQE(m_scintPhotonScaleWeight>1.);
    // scint->SetPreQE(1./m_scintPhotonScaleWeight);
    scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); //1.);
    // scint->SetUseFastMu300nsTrick(m_useFastMu300nsTrick);
    scint->SetTrackSecondariesFirst(true);
    // if (!m_useScintillation) scint->SetNoOp();
#  endif
#else  // standard G4 scint
    G4Scintillation* scint = 0;
    if (m_useScintillation) 
    {
        LOG(info) << "Using standard G4Scintillation." ;
        scint = new G4Scintillation();
        scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); // 1.);
        scint->SetTrackSecondariesFirst(true);
    }
#endif
    fScintillationProcess = scint ; 



    G4OpAbsorption* absorb  = m_useAbsorption ? new G4OpAbsorption() : NULL ;


#ifdef USE_POWER_THIRD_RAYLEIGH
    DsG4OpRayleigh* rayleigh = m_useRayleigh  ? new DsG4OpRayleigh() : NULL ; 
#else
    OpRayleigh* rayleigh = m_useRayleigh  ? new OpRayleigh() : NULL ; 
#endif

    //G4OpBoundaryProcess* boundproc = new G4OpBoundaryProcess();
    DsG4OpBoundaryProcess* boundproc = new DsG4OpBoundaryProcess(m_g4);
    boundproc->SetModel(unified);

    //G4FastSimulationManagerProcess* fast_sim_man = new G4FastSimulationManagerProcess("fast_sim_man");
    
    theParticleIterator->reset();
    while( (*theParticleIterator)() ) {

        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
    
        // Caution: as of G4.9, Cerenkov becomes a Discrete Process.
        // This code assumes a version of G4Cerenkov from before this version.
        //
        /// SCB: Contrary to above FUD-comment, contemporary G4 code such as 
        ///      OpNovicePhysicsList sets up Cerenkov just like this

        if(cerenkov && cerenkov->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(cerenkov);
            pmanager->SetProcessOrdering(cerenkov, idxPostStep);
            LOG(debug) << "Process: adding Cherenkov to " 
                       << particle->GetParticleName() ;
        }

/*
        if(scint && scint->IsApplicable(*particle))
        {
            pmanager->AddProcess(scint);
            pmanager->SetProcessOrderingToLast(scint, idxAtRest);
            pmanager->SetProcessOrderingToLast(scint, idxPostStep);
            LOG(debug) << "Process: adding Scintillation to "
                       << particle->GetParticleName() ;
        }
*/

        if(particle == G4OpticalPhoton::Definition()) 
        {
            if(absorb) pmanager->AddDiscreteProcess(absorb);
            if(rayleigh) pmanager->AddDiscreteProcess(rayleigh);
            pmanager->AddDiscreteProcess(boundproc);
            //pmanager->AddDiscreteProcess(fast_sim_man);
        }
    }
}



/*
void OpNovicePhysicsList::ConstructOpNovice()
{
  fCerenkovProcess = new Cerenkov("Cerenkov");
  fCerenkovProcess->SetMaxNumPhotonsPerStep(fMaxNumPhotonStep);
  fCerenkovProcess->SetMaxBetaChangePerStep(10.0);
  fCerenkovProcess->SetTrackSecondariesFirst(true);

  //fScintillationProcess = new G4Scintillation("Scintillation");
  fScintillationProcess = new Scintillation("Scintillation");
  fScintillationProcess->SetScintillationYieldFactor(1.);
  fScintillationProcess->SetTrackSecondariesFirst(true);
  fAbsorptionProcess = new G4OpAbsorption();

  //fRayleighScatteringProcess = new G4OpRayleigh();
  fRayleighScatteringProcess = new OpRayleigh();

  fMieHGScatteringProcess = new G4OpMieHG();
  fBoundaryProcess = new G4OpBoundaryProcess();

  if(fCerenkovProcess) 
  fCerenkovProcess->SetVerboseLevel(fVerboseLevel);

  fScintillationProcess->SetVerboseLevel(fVerboseLevel);
  fAbsorptionProcess->SetVerboseLevel(fVerboseLevel);
  fRayleighScatteringProcess->SetVerboseLevel(fVerboseLevel);
  fMieHGScatteringProcess->SetVerboseLevel(fVerboseLevel);
  fBoundaryProcess->SetVerboseLevel(fVerboseLevel);
  
  // Use Birks Correction in the Scintillation process
  if(G4Threading::IsMasterThread())
  {
    G4EmSaturation* emSaturation =
              G4LossTableManager::Instance()->EmSaturation();
      fScintillationProcess->AddSaturation(emSaturation);
  }

  theParticleIterator->reset();
  while( (*theParticleIterator)() ){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();
    if (fCerenkovProcess && fCerenkovProcess->IsApplicable(*particle)) {
      pmanager->AddProcess(fCerenkovProcess);
      pmanager->SetProcessOrdering(fCerenkovProcess,idxPostStep);
    }
    if (fScintillationProcess->IsApplicable(*particle)) {
      pmanager->AddProcess(fScintillationProcess);
      pmanager->SetProcessOrderingToLast(fScintillationProcess, idxAtRest);
      pmanager->SetProcessOrderingToLast(fScintillationProcess, idxPostStep);
    }
    if (particleName == "opticalphoton") {
      G4cout << " AddDiscreteProcess to OpticalPhoton " << G4endl;
      pmanager->AddDiscreteProcess(fAbsorptionProcess);
      pmanager->AddDiscreteProcess(fRayleighScatteringProcess);
      pmanager->AddDiscreteProcess(fMieHGScatteringProcess);
      pmanager->AddDiscreteProcess(fBoundaryProcess);
    }
  }
}

*/






OpNovicePhysicsList::~OpNovicePhysicsList() { delete fMessenger; }


void OpNovicePhysicsList::ConstructParticle()
{
  // In this method, static member functions should be called
  // for all particles which you want to use.
  // This ensures that objects of these particle types will be
  // created in the program.

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
}







#include "G4Decay.hh"


void OpNovicePhysicsList::ConstructDecay()
{
  // Add Decay Process
  G4Decay* theDecayProcess = new G4Decay();
  theParticleIterator->reset();
  while( (*theParticleIterator)() ){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    if (theDecayProcess->IsApplicable(*particle)) {
      pmanager ->AddProcess(theDecayProcess);
      // set ordering for PostStepDoIt and AtRestDoIt
      pmanager ->SetProcessOrdering(theDecayProcess, idxPostStep);
      pmanager ->SetProcessOrdering(theDecayProcess, idxAtRest);
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




void OpNovicePhysicsList::Summary(const char* msg)
{
    LOG(info) << msg ; 
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
         G4ParticleDefinition* particle = theParticleIterator->value();
         G4String particleName = particle->GetParticleName();

         G4ProcessManager* pmanager = particle->GetProcessManager();

         unsigned int npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
         LOG(info) << particleName << " " << npro ;

         if(!pmanager) continue ;  

         G4ProcessVector* procs = pmanager->GetProcessList();
         for(unsigned int i=0 ; i < npro ; i++)
         {
             G4VProcess* proc = (*procs)[i] ; 
             LOG(info) << std::setw(3) << i << proc->GetProcessName()  ;
         }
    }
}

void OpNovicePhysicsList::dumpProcesses(const char* msg)
{
    LOG(info) << msg << " size " << m_procs.size() ; 
    typedef std::set<G4VProcess*>::const_iterator SPI ;
    for(SPI it=m_procs.begin() ; it != m_procs.end() ; it++)
    { 
        G4VProcess* proc = *it ; 
        LOG(info)
             << std::setw(25) << proc->GetProcessName()
             << std::setw(4) << proc->GetVerboseLevel()
             ; 
    }
}


void  OpNovicePhysicsList::setupEmVerbosity(unsigned int verbosity)
{
   // these are used in Em process constructors, so do this prior to process creation
    G4EmParameters* empar = G4EmParameters::Instance() ;
    empar->SetVerbose(verbosity); 
    empar->SetWorkerVerbose(verbosity); 
}


void  OpNovicePhysicsList::setProcessVerbosity(int verbosity)
{
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
         G4ParticleDefinition* particle = theParticleIterator->value();
         G4String particleName = particle->GetParticleName();
         G4ProcessManager* pmanager = particle->GetProcessManager();
         if(!pmanager) continue ; 

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
                 LOG(debug) << "OpNovicePhysicsList::setProcessVerbosity " << particleName << ":" << processName << " from " << prior << " to " << proc->GetVerboseLevel() ;

         }
   } 
}


void OpNovicePhysicsList::collectProcesses()
{
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
         G4ParticleDefinition* particle = theParticleIterator->value();
         G4String particleName = particle->GetParticleName();
         G4ProcessManager* pmanager = particle->GetProcessManager();
         if(!pmanager) 
         {
            LOG(info) << "OpNovicePhysicsList::collectProcesses no ProcessManager for " << particleName ;
            continue ;  
         }

         unsigned npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
         G4ProcessVector* procs = pmanager->GetProcessList();
         for(unsigned int i=0 ; i < npro ; i++)
         {
             G4VProcess* proc = (*procs)[i] ; 
             m_procs.insert(proc);
             m_procl.push_back(proc);
         }
    }
}





void OpNovicePhysicsList::ConstructEM()
{
  theParticleIterator->reset();
  while( (*theParticleIterator)() ){
    G4ParticleDefinition* particle = theParticleIterator->value();
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

#include "G4Threading.hh"


void OpNovicePhysicsList::SetVerbose(G4int verbose)
{
  fVerboseLevel = verbose;

  if(fCerenkovProcess)
  fCerenkovProcess->SetVerboseLevel(fVerboseLevel);

  fScintillationProcess->SetVerboseLevel(fVerboseLevel);
  fAbsorptionProcess->SetVerboseLevel(fVerboseLevel);
  fRayleighScatteringProcess->SetVerboseLevel(fVerboseLevel);
  fMieHGScatteringProcess->SetVerboseLevel(fVerboseLevel);
  fBoundaryProcess->SetVerboseLevel(fVerboseLevel);
}


void OpNovicePhysicsList::SetNbOfPhotonsCerenkov(G4int MaxNumber)
{
  fMaxNumPhotonStep = MaxNumber;

  if(fCerenkovProcess) 
  fCerenkovProcess->SetMaxNumPhotonsPerStep(fMaxNumPhotonStep);
}


void OpNovicePhysicsList::SetCuts()
{
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
  //   the default cut value for all particle types
  //
  SetCutsWithDefault();

  //if (verboseLevel>0) DumpCutValuesTable();
  //   


}


void OpNovicePhysicsList::dump(const char* msg)
{
    dumpParam(msg);
    dumpMaterials(msg);
    dumpRayleigh(msg);
}

void OpNovicePhysicsList::dumpMaterials(const char* msg)
{
    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    const G4int numOfMaterials = G4Material::GetNumberOfMaterials();

    LOG(info) << msg 
              << " numOfMaterials " << numOfMaterials
              ;

    
    for(int i=0 ; i < numOfMaterials ; i++)
    {
        G4Material* material = (*theMaterialTable)[i];
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

        CMPT cmpt(mpt);
        LOG(debug) << msg << cmpt.description(material->GetName().c_str()) ; 

    }


}

void OpNovicePhysicsList::dumpRayleigh(const char* msg)
{
    LOG(info) << msg ; 
    if(fRayleighScatteringProcess)
    {
        G4PhysicsTable* ptab = fRayleighScatteringProcess->GetPhysicsTable();
        if(ptab) 
            fRayleighScatteringProcess->DumpPhysicsTable() ;    
        else
            LOG(debug) << "OpNovicePhysicsList::dumpRayleigh no physics table"   ;
    }

}






