#include "SteppingAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"

#include <cstdio>


SteppingAction::SteppingAction(RecorderBase* recorder)
   : 
   G4UserSteppingAction(),
   m_recorder(recorder)
{ 
  fScintillationCounter = 0;
  fCerenkovCounter      = 0;
  fEventNumber = -1;
}


SteppingAction::~SteppingAction()
{ ; }


void SteppingAction::UserSteppingAction(const G4Step* step)
{

  G4Track* track = step->GetTrack();
  G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

  G4int stepNum = track->GetCurrentStepNumber() ;

  G4int eventNumber = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  if (eventNumber != fEventNumber) {
   
     G4cout << "SteppingAction::UserSteppingAction evt " << eventNumber << " : " << particleName << G4endl ;
     fEventNumber = eventNumber;
     fScintillationCounter = 0;
     fCerenkovCounter = 0;
  }

  G4StepPoint* pre  = step->GetPreStepPoint() ;
  G4StepPoint* post = step->GetPostStepPoint() ;

  const G4ThreeVector& pre_pos = pre->GetPosition();
  const G4ThreeVector& post_pos = post->GetPosition();

 if(stepNum % 10 == 0)
 G4cout << std::setw( 5) << "#Step#"     << " "
        << std::setw( 6) << "X"          << "    "
        << std::setw( 6) << "Y"          << "    "
        << std::setw( 6) << "Z"          << "    "
        << std::setw( 9) << "KineE"      << " "
        << std::setw( 9) << "dEStep"     << " "
        << std::setw(10) << "StepLeng"
        << std::setw(10) << "TrakLeng"
        << std::setw(10) << "Volume"    << "  "
        << std::setw(10) << "Process"   << G4endl;

  G4cout << std::setw(5) << stepNum << " "
    << std::setw(16) << G4BestUnit(track->GetPosition().x(),"Length")
    << std::setw(16) << G4BestUnit(track->GetPosition().y(),"Length")
    << std::setw(16) << G4BestUnit(track->GetPosition().z(),"Length")
    << std::setw(16) << G4BestUnit(track->GetKineticEnergy(),"Energy")
    << std::setw(16) << G4BestUnit(step->GetTotalEnergyDeposit(),"Energy")
    << std::setw(16) << G4BestUnit(step->GetStepLength(),"Length")
    << std::setw(16) << G4BestUnit(track->GetTrackLength(),"Length")
    << "  " << G4endl ;


  
  //if (particleName == "opticalphoton") return;

  const std::vector<const G4Track*>* secondaries = step->GetSecondaryInCurrentStep();


  G4cout << "secondaries " << secondaries->size() << G4endl ; 

  if (secondaries->size()>0) 
  {
     for(unsigned int i=0; i<secondaries->size(); ++i) 
     {
        if (secondaries->at(i)->GetParentID()>0) 
        {
           if(secondaries->at(i)->GetDynamicParticle()->GetParticleDefinition() == G4OpticalPhoton::OpticalPhotonDefinition())
           {
              if (secondaries->at(i)->GetCreatorProcess()->GetProcessName() == "Scintillation") fScintillationCounter++;
              if (secondaries->at(i)->GetCreatorProcess()->GetProcessName() == "Cerenkov")      fCerenkovCounter++;
           }
        }
     }
  }




}

