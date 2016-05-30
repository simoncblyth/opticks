//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "globals.hh"
#include "OpNovicePhysicsList.hh"
#include "OpNovicePhysicsListMessenger.hh"
#include "CMPT.hh"


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

#include "G4Cerenkov.hh"
//#include "G4Scintillation.hh"
#include "Scintillation.hh"
#include "G4OpAbsorption.hh"

//#include "G4OpRayleigh.hh"
#include "OpRayleigh.hh"

#include "G4OpMieHG.hh"
#include "G4OpBoundaryProcess.hh"

#include "G4LossTableManager.hh"
#include "G4EmSaturation.hh"

G4ThreadLocal G4int OpNovicePhysicsList::fVerboseLevel = 1;
G4ThreadLocal G4int OpNovicePhysicsList::fMaxNumPhotonStep = 20;
G4ThreadLocal G4Cerenkov* OpNovicePhysicsList::fCerenkovProcess = 0;

//G4ThreadLocal G4Scintillation* OpNovicePhysicsList::fScintillationProcess = 0;
G4ThreadLocal Scintillation* OpNovicePhysicsList::fScintillationProcess = 0;

G4ThreadLocal G4OpAbsorption* OpNovicePhysicsList::fAbsorptionProcess = 0;

//G4ThreadLocal G4OpRayleigh* OpNovicePhysicsList::fRayleighScatteringProcess = 0;
G4ThreadLocal OpRayleigh* OpNovicePhysicsList::fRayleighScatteringProcess = 0;

G4ThreadLocal G4OpMieHG* OpNovicePhysicsList::fMieHGScatteringProcess = 0;
G4ThreadLocal G4OpBoundaryProcess* OpNovicePhysicsList::fBoundaryProcess = 0;
 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNovicePhysicsList::OpNovicePhysicsList() 
 : G4VUserPhysicsList()
{
  fMessenger = new OpNovicePhysicsListMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNovicePhysicsList::~OpNovicePhysicsList() { delete fMessenger; }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......






void OpNovicePhysicsList::ConstructProcess()
{
  setupEmVerbosity(0); 

  AddTransportation();
  ConstructDecay();
  ConstructEM();
  ConstructOp();

  dump("OpNovicePhysicsList::ConstructProcess"); 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "G4Decay.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


#include "NLog.hpp"

void OpNovicePhysicsList::Summary(const char* msg)
{
    LOG(info) << msg ; 
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
         G4ParticleDefinition* particle = theParticleIterator->value();
         G4String particleName = particle->GetParticleName();

         G4ProcessManager* pmanager = particle->GetProcessManager();

         int npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
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


void  OpNovicePhysicsList::setProcessVerbosity(unsigned int verbosity)
{
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
         G4ParticleDefinition* particle = theParticleIterator->value();
         G4String particleName = particle->GetParticleName();
         G4ProcessManager* pmanager = particle->GetProcessManager();
         if(!pmanager) continue ; 

         int npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
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

         int npro = pmanager ? pmanager->GetProcessListLength() : 0 ;
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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
#include "G4Threading.hh"

void OpNovicePhysicsList::ConstructOp()
{
  fCerenkovProcess = new G4Cerenkov("Cerenkov");
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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void OpNovicePhysicsList::SetNbOfPhotonsCerenkov(G4int MaxNumber)
{
  fMaxNumPhotonStep = MaxNumber;

  if(fCerenkovProcess) 
  fCerenkovProcess->SetMaxNumPhotonsPerStep(fMaxNumPhotonStep);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

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

    
    for(unsigned int i=0 ; i < numOfMaterials ; i++)
    {
        G4Material* material = (*theMaterialTable)[i];
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

        CMPT cmpt(mpt);
        LOG(info) << msg << cmpt.description(material->GetName().c_str()) ; 

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
            LOG(info) << "OpNovicePhysicsList::dumpRayleigh no physics table"   ;
    }

}


