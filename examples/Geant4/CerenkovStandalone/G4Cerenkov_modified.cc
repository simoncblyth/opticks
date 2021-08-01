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
// $Id: G4Cerenkov.cc 108508 2018-02-15 15:54:35Z gcosmo $
//
////////////////////////////////////////////////////////////////////////
// Cerenkov Radiation Class Implementation
////////////////////////////////////////////////////////////////////////
//
// File:        G4Cerenkov.cc
// Description: Discrete Process -- Generation of Cerenkov Photons
// Version:     2.1
// Created:     1996-02-21
// Author:      Juliet Armstrong
// Updated:     2007-09-30 by Peter Gumplinger
//              > change inheritance to G4VDiscreteProcess
//              GetContinuousStepLimit -> GetMeanFreePath (StronglyForced)
//              AlongStepDoIt -> PostStepDoIt
//              2005-08-17 by Peter Gumplinger
//              > change variable name MeanNumPhotons -> MeanNumberOfPhotons
//              2005-07-28 by Peter Gumplinger
//              > add G4ProcessType to constructor
//              2001-09-17, migration of Materials to pure STL (mma)
//              2000-11-12 by Peter Gumplinger
//              > add check on CerenkovAngleIntegrals->IsFilledVectorExist()
//              in method GetAverageNumberOfPhotons
//              > and a test for MeanNumberOfPhotons <= 0.0 in DoIt
//              2000-09-18 by Peter Gumplinger
//              > change: aSecondaryPosition=x0+rand*aStep.GetDeltaPosition();
//                        aSecondaryTrack->SetTouchable(0);
//              1999-10-29 by Peter Gumplinger
//              > change: == into <= in GetContinuousStepLimit
//              1997-08-08 by Peter Gumplinger
//              > add protection against /0
//              > G4MaterialPropertiesTable; new physics/tracking scheme
//
// mail:        gum@triumf.ca
//
////////////////////////////////////////////////////////////////////////

#include "G4ios.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Poisson.hh"
#include "G4EmProcessSubType.hh"

#include "G4LossTableManager.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleDefinition.hh"

#include "G4Cerenkov_modified.hh"

#ifdef INSTRUMENTED
#include "OpticksDebug.hh"
#include "OpticksRandom.hh"
#endif


/////////////////////////
// Class Implementation  
/////////////////////////

//G4bool G4Cerenkov::fTrackSecondariesFirst = false;
//G4double G4Cerenkov::fMaxBetaChange = 0.;
//G4int G4Cerenkov::fMaxPhotons = 0;

  //////////////
  // Operators
  //////////////

// G4Cerenkov::operator=(const G4Cerenkov &right)
// {
// }

  /////////////////
  // Constructors
  /////////////////

G4Cerenkov_modified::G4Cerenkov_modified(const G4String& processName, G4ProcessType type)
           : G4VProcess(processName, type),
             fTrackSecondariesFirst(false),
             fMaxBetaChange(0.0),
             fMaxPhotons(0),
             fStackingFlag(true),
#ifdef INSTRUMENTED
             override_fNumPhotons(0),
#endif
             fNumPhotons(0)
{
  SetProcessSubType(fCerenkov);

  thePhysicsTable = nullptr;

  if (verboseLevel>0) {
     G4cout << GetProcessName() << " is created " << G4endl;
  }
}

// G4Cerenkov::G4Cerenkov(const G4Cerenkov &right)
// {
// }

  ////////////////
  // Destructors
  ////////////////

G4Cerenkov_modified::~G4Cerenkov_modified()
{
  if (thePhysicsTable != nullptr) {
     thePhysicsTable->clearAndDestroy();
     delete thePhysicsTable;
  }
}

  ////////////
  // Methods
  ////////////

G4bool G4Cerenkov_modified::IsApplicable(const G4ParticleDefinition& aParticleType)
{
  G4bool result = false;

  if (aParticleType.GetPDGCharge() != 0.0 &&
      aParticleType.GetPDGMass() != 0.0 &&
      aParticleType.GetParticleName() != "chargedgeantino" &&
      !aParticleType.IsShortLived() ) { result = true; }

  return result;
}

void G4Cerenkov_modified::SetTrackSecondariesFirst(const G4bool state)
{
  fTrackSecondariesFirst = state;
}

void G4Cerenkov_modified::SetMaxBetaChangePerStep(const G4double value)
{
  fMaxBetaChange = value*CLHEP::perCent;
}

void G4Cerenkov_modified::SetMaxNumPhotonsPerStep(const G4int NumPhotons)
{
  fMaxPhotons = NumPhotons;
}

void G4Cerenkov_modified::BuildPhysicsTable(const G4ParticleDefinition&)
{
  if (!thePhysicsTable) BuildThePhysicsTable();
}

// PostStepDoIt
// -------------
//
G4VParticleChange* G4Cerenkov_modified::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)

// This routine is called for each tracking Step of a charged particle
// in a radiator. A Poisson-distributed number of photons is generated
// according to the Cerenkov formula, distributed evenly along the track
// segment and uniformly azimuth w.r.t. the particle direction. The
// parameters are then transformed into the Master Reference System, and
// they are added to the particle change.

{
  ////////////////////////////////////////////////////
  // Should we ensure that the material is dispersive?
  ////////////////////////////////////////////////////


  aParticleChange.Initialize(aTrack);

  const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
  const G4Material* aMaterial = aTrack.GetMaterial();

  G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
  G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();

  G4ThreeVector x0 = pPreStepPoint->GetPosition();
  G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
  G4double t0 = pPreStepPoint->GetGlobalTime();

  G4MaterialPropertiesTable* aMaterialPropertiesTable =
                               aMaterial->GetMaterialPropertiesTable();
  if (!aMaterialPropertiesTable) return pParticleChange;

  G4MaterialPropertyVector* Rindex = 
                aMaterialPropertiesTable->GetProperty(kRINDEX); 
  if (!Rindex) return pParticleChange;

  // particle charge
  G4double charge = aParticle->GetDefinition()->GetPDGCharge();

  // particle beta
  G4double beta = (pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta())*0.5;

  fNumPhotons = 0;

  G4double MeanNumberOfPhotons = 
                     GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

  if (MeanNumberOfPhotons <= 0.0) {

     // return unchanged particle and no secondaries

     aParticleChange.SetNumberOfSecondaries(0);
 
     return pParticleChange;

  }

  G4double step_length = aStep.GetStepLength();

  MeanNumberOfPhotons = MeanNumberOfPhotons * step_length;

  fNumPhotons = (G4int) G4Poisson(MeanNumberOfPhotons);

#ifdef INSTRUMENTED
   if( override_fNumPhotons > 0 )
   { 
       fNumPhotons = override_fNumPhotons ; 
   }
#endif


  // calculate the fNumPhotons1 and fNumPhotons2 {

  // }


  // if ( fNumPhotons <= 0 || !fStackingFlag ) {
  if ( fNumPhotons <= 0 ) {

     // return unchanged particle and no secondaries  

     aParticleChange.SetNumberOfSecondaries(0);

     fNumPhotons1 = 0;
     fNumPhotons2 = 0;

     return pParticleChange;

  }

  ////////////////////////////////////////////////////////////////
  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
  G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
  G4double dp = Pmax - Pmin;


#ifdef FLOAT_TEST
  float nMax = Rindex->GetMaxValue();
#else
  G4double nMax = Rindex->GetMaxValue();
#endif
  if (Rindex) {
      // nMax = Rindex->GetMaxValue();
      size_t ri_sz = Rindex->GetVectorLength();
   
      for (size_t i = 0; i < ri_sz; ++i) {
          if ((*Rindex)[i]>nMax) nMax = (*Rindex)[i];
      }
  }

  G4double BetaInverse = 1./beta;

#ifdef FLOAT_TEST
  float maxCos = BetaInverse / nMax; 
  float maxSin2 = (1.f - maxCos) * (1.f + maxCos);
#else
  G4double maxCos = BetaInverse / nMax; 
  G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
#endif

  G4double beta1 = pPreStepPoint ->GetBeta();
  G4double beta2 = pPostStepPoint->GetBeta();

  G4double MeanNumberOfPhotons1 =
                     GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
  G4double MeanNumberOfPhotons2 =
                     GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);

  fNumPhotons1 = MeanNumberOfPhotons1;
  fNumPhotons2 = MeanNumberOfPhotons2;


#ifdef INSTRUMENTED
  par->append( BetaInverse, "BetaInverse" );  
  par->append( beta       , "beta" );  
  par->append( Pmin       , "Pmin" );  
  par->append( Pmax       , "Pmax" );  

  par->append( nMax       , "nMax" );  
  par->append( maxCos     , "maxCos" );  
  par->append( maxSin2    , "maxSin2" );  
  par->append( fNumPhotons, "fNumPhotons" );  
#endif



  if (MeanNumberOfPhotons1 <= 0.0 or MeanNumberOfPhotons2<=0.0) {

     // return unchanged particle and no secondaries

     aParticleChange.SetNumberOfSecondaries(0);
 
     // Force no secondaries
     fNumPhotons = 0;
     fNumPhotons1 = 0;
     fNumPhotons2 = 0;

     return pParticleChange;

  }

  // if stacking is false, then stop the generation of Cerenkov photons
  if (!fStackingFlag) {
     aParticleChange.SetNumberOfSecondaries(0);
 
     return pParticleChange;
  }

  ////////////////////////////////////////////////////////////////

  aParticleChange.SetNumberOfSecondaries(fNumPhotons);

  if (fTrackSecondariesFirst) {
     if (aTrack.GetTrackStatus() == fAlive )
         aParticleChange.ProposeTrackStatus(fSuspend);
  }


  for (G4int i = 0; i < fNumPhotons; i++) {
      // Determine photon energy

#ifdef FLOAT_TEST
      float rand, rand0, rand1 ;
      float sampledEnergy, sampledRI; 
      float cosTheta, sin2Theta;
#else
      G4double rand, rand0, rand1 ;
      G4double sampledEnergy, sampledRI; 
      G4double cosTheta, sin2Theta;
#endif

#ifdef INSTRUMENTED
      unsigned head_count = 0 ; 
      unsigned tail_count = 0 ; 
      unsigned continue_count = 0 ; 
      unsigned condition_count = 0 ;
      int seqidx = -1 ;  
      if(rnd)
      {
          rnd->setSequenceIndex(i); 
          seqidx = rnd->getSequenceIndex(); 

          if(i < 10) std::cout 
              << " i " << std::setw(6) << i 
              << " seqidx " << std::setw(7) << seqidx
              << " Pmin/eV " << std::fixed << std::setw(10) << std::setprecision(5) << Pmin/eV
              << " Pmax/eV " << std::fixed << std::setw(10) << std::setprecision(5) << Pmax/eV
              << " dp/eV " << std::fixed << std::setw(10) << std::setprecision(5) << dp/eV
              << " maxSin2 "  << std::fixed << std::setw(10) << std::setprecision(5) << maxSin2
              << std::endl 
              ;

      }
#endif

      // sample an energy

      do {
#ifdef INSTRUMENTED
         head_count += 1 ; 
#endif
         rand0 = G4UniformRand();  
         sampledEnergy = Pmin + rand0 * dp; 
         sampledRI = Rindex->Value(sampledEnergy);


#ifdef SKIP_CONTINUE
#else
         // check if n(E) > 1/beta
         if (sampledRI < BetaInverse) {
#ifdef INSTRUMENTED
             continue_count += 1 ; 
#endif
             continue;
         }

#endif

#ifdef INSTRUMENTED
         tail_count += 1 ; 
#endif
 

         cosTheta = BetaInverse / sampledRI;  

         // G4cout << "TAO --> L" << __LINE__ << ": " 
         //        << " BetaInverse : " << BetaInverse
         //        << " sampledRI : " << sampledRI
         //        << " cosTheta : " << cosTheta
         //        << G4endl;

#ifdef FLOAT_TEST
         sin2Theta = (1.f - cosTheta)*(1.f + cosTheta);
#else
         sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
#endif

         rand1 = G4UniformRand();  
#ifdef ONE_RAND
         rand1 = 1.0 ; 
#endif

        // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
#ifdef INSTRUMENTED

         if( i < 10 ) std::cout 
             << " tc " << std::setw(6) << tail_count 
             << " u0 " << std::fixed << std::setw(10) << std::setprecision(5) << rand0 
             << " eV " << std::fixed << std::setw(10) << std::setprecision(5) << sampledEnergy/eV
             << " ri " << std::fixed << std::setw(10) << std::setprecision(5) << sampledRI
             << " ct " << std::fixed << std::setw(10) << std::setprecision(5) << cosTheta
             << " s2 " << std::fixed << std::setw(10) << std::setprecision(5) << sin2Theta
             << " rand1*maxSin2 " << std::fixed << std::setw(10) << std::setprecision(5) << rand1*maxSin2
             << " rand1*maxSin2 - sin2Theta " <<  std::fixed << std::setw(10) << std::setprecision(5) << rand1*maxSin2 - sin2Theta
             << " loop " << ( rand1*maxSin2 > sin2Theta ? "Y" : "N" )
             << std::endl 
             ; 


      } while ( looping_condition(condition_count) && rand1*maxSin2 > sin2Theta  );
#else
      } while (rand1*maxSin2 > sin2Theta);
#endif

#ifdef INSTRUMENTED
        G4double sampledEnergy_eV = sampledEnergy/eV ; 
        G4double sampledWavelength_nm = h_Planck*c_light/sampledEnergy/nm ;

        gen->append( sampledEnergy_eV ,       "sampledEnergy" ); 
        gen->append( sampledWavelength_nm ,    "sampledWavelength" ); 
        gen->append( sampledRI ,               "sampledRI" ); 
        gen->append( cosTheta ,                "cosTheta" ); 

        gen->append( sin2Theta ,               "sin2Theta" ); 
        gen->append( head_count ,     tail_count,       "head_tail" ); 
        gen->append( continue_count , condition_count,  "continue_condition" ); 
        gen->append( BetaInverse , "BetaInverse" ); 
 
        if(rnd)
        {
           rnd->setSequenceIndex(-1); 
        }
#endif
 



      // Generate random position of photon on cone surface 
      // defined by Theta 

      rand = G4UniformRand();

      G4double phi = twopi*rand;
      G4double sinPhi = std::sin(phi);
      G4double cosPhi = std::cos(phi);

      // calculate x,y, and z components of photon energy
      // (in coord system with primary particle direction 
      //  aligned with the z axis)

      G4double sinTheta = std::sqrt(sin2Theta); 
      G4double px = sinTheta*cosPhi;
      G4double py = sinTheta*sinPhi;
      G4double pz = cosTheta;

      // Create photon momentum direction vector 
      // The momentum direction is still with respect
      // to the coordinate system where the primary
      // particle direction is aligned with the z axis  

      G4ParticleMomentum photonMomentum(px, py, pz);

      // Rotate momentum direction back to global reference
      // system 

      photonMomentum.rotateUz(p0);

      // Determine polarization of new photon 

      G4double sx = cosTheta*cosPhi;
      G4double sy = cosTheta*sinPhi; 
      G4double sz = -sinTheta;

      G4ThreeVector photonPolarization(sx, sy, sz);

      // Rotate back to original coord system 

      photonPolarization.rotateUz(p0);

      // Generate a new photon:

      G4DynamicParticle* aCerenkovPhoton =
        new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(),photonMomentum);

      aCerenkovPhoton->SetPolarization(photonPolarization.x(),
                                       photonPolarization.y(),
                                       photonPolarization.z());

      aCerenkovPhoton->SetKineticEnergy(sampledEnergy);

      // Generate new G4Track object:

      G4double NumberOfPhotons, N;

      do {
         rand = G4UniformRand();
         NumberOfPhotons = MeanNumberOfPhotons1 - rand *
                                (MeanNumberOfPhotons1-MeanNumberOfPhotons2);
         N = G4UniformRand() *
                        std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);
        // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
      } while (N > NumberOfPhotons);

      G4double delta = rand * aStep.GetStepLength();

      G4double deltaTime = delta / (pPreStepPoint->GetVelocity()+
                                      rand*(pPostStepPoint->GetVelocity()-
                                            pPreStepPoint->GetVelocity())*0.5);

      G4double aSecondaryTime = t0 + deltaTime;

      G4ThreeVector aSecondaryPosition = x0 + rand * aStep.GetDeltaPosition();

      G4Track* aSecondaryTrack = 
               new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);

      aSecondaryTrack->SetTouchableHandle(
                               aStep.GetPreStepPoint()->GetTouchableHandle());

      aSecondaryTrack->SetParentID(aTrack.GetTrackID());

      aParticleChange.AddSecondary(aSecondaryTrack);

  }

  if (verboseLevel>0) {
     G4cout <<"\n Exiting from G4Cerenkov_modified::DoIt -- NumberOfSecondaries = "
      << aParticleChange.GetNumberOfSecondaries() << G4endl;
  }

  return pParticleChange;
}


#ifdef INSTRUMENTED
bool G4Cerenkov_modified::looping_condition(unsigned& count)
{   
    count += 1 ;
    return true ;
}   


#endif


// BuildThePhysicsTable for the Cerenkov process
// ---------------------------------------------
//

void G4Cerenkov_modified::BuildThePhysicsTable()
{
  if (thePhysicsTable) return;

  const G4MaterialTable* theMaterialTable=
  G4Material::GetMaterialTable();
  G4int numOfMaterials = G4Material::GetNumberOfMaterials();

  // create new physics table
  
  thePhysicsTable = new G4PhysicsTable(numOfMaterials);

  // loop for materials

  for (G4int i=0 ; i < numOfMaterials; i++) {

      G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector = 0;

      // Retrieve vector of refraction indices for the material
      // from the material's optical properties table 

      G4Material* aMaterial = (*theMaterialTable)[i];

      G4MaterialPropertiesTable* aMaterialPropertiesTable =
                                      aMaterial->GetMaterialPropertiesTable();

      if (aMaterialPropertiesTable) {
         aPhysicsOrderedFreeVector = new G4PhysicsOrderedFreeVector();
         G4MaterialPropertyVector* theRefractionIndexVector = 
                              aMaterialPropertiesTable->GetProperty(kRINDEX);

         if (theRefractionIndexVector) {

            // Retrieve the first refraction index in vector
            // of (photon energy, refraction index) pairs 

            G4double currentRI = (*theRefractionIndexVector)[0];

            if (currentRI > 1.0) {

               // Create first (photon energy, Cerenkov Integral)
               // pair  

               G4double currentPM = theRefractionIndexVector->Energy(0);
               G4double currentCAI = 0.0;

               aPhysicsOrderedFreeVector->InsertValues(currentPM , currentCAI);

               // Set previous values to current ones prior to loop

               G4double prevPM  = currentPM;
               G4double prevCAI = currentCAI;
               G4double prevRI  = currentRI;

               // loop over all (photon energy, refraction index)
               // pairs stored for this material  

               for (size_t ii = 1;
                           ii < theRefractionIndexVector->GetVectorLength();
                           ++ii) {
                   currentRI = (*theRefractionIndexVector)[ii];
                   currentPM = theRefractionIndexVector->Energy(ii);

                   currentCAI = 0.5*(1.0/(prevRI*prevRI) +
                                     1.0/(currentRI*currentRI));

                   currentCAI = prevCAI + (currentPM - prevPM) * currentCAI;

                   aPhysicsOrderedFreeVector->
                                         InsertValues(currentPM, currentCAI);

                   prevPM  = currentPM;
                   prevCAI = currentCAI;
                   prevRI  = currentRI;
               }

            }
         }
      }

      // The Cerenkov integral for a given material
      // will be inserted in thePhysicsTable
      // according to the position of the material in
      // the material table. 

      thePhysicsTable->insertAt(i,aPhysicsOrderedFreeVector); 

  }
}

// GetMeanFreePath
// ---------------
//

G4double G4Cerenkov_modified::GetMeanFreePath(const G4Track&,
                                           G4double,
                                           G4ForceCondition*)
{
  return 1.;
}

G4double G4Cerenkov_modified::PostStepGetPhysicalInteractionLength(
                                           const G4Track& aTrack,
                                           G4double,
                                           G4ForceCondition* condition)
{
  *condition = NotForced;
  G4double StepLimit = DBL_MAX;

  const G4Material* aMaterial = aTrack.GetMaterial();
  G4int materialIndex = aMaterial->GetIndex();

  // If Physics Vector is not defined no Cerenkov photons
  //    this check avoid string comparison below

  if(!(*thePhysicsTable)[materialIndex]) { return StepLimit; }

  const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
  const G4MaterialCutsCouple* couple = aTrack.GetMaterialCutsCouple();

  G4double kineticEnergy = aParticle->GetKineticEnergy();
  const G4ParticleDefinition* particleType = aParticle->GetDefinition();
  G4double mass = particleType->GetPDGMass();

  // particle beta
  G4double beta = aParticle->GetTotalMomentum() /
                  aParticle->GetTotalEnergy();
  // particle gamma
  G4double gamma = aParticle->GetTotalEnergy()/mass;


  G4MaterialPropertiesTable* aMaterialPropertiesTable =
                                      aMaterial->GetMaterialPropertiesTable();

  G4MaterialPropertyVector* Rindex = NULL;

  if (aMaterialPropertiesTable)
                     Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX);

  G4double nMax;
  if (Rindex) {
      // nMax = Rindex->GetMaxValue();
      size_t ri_sz = Rindex->GetVectorLength();
      nMax = (*Rindex)[0];
      for (size_t i = 1; i < ri_sz; ++i) {
          if ((*Rindex)[i]>nMax) nMax = (*Rindex)[i];
      }
  } else {
     return StepLimit;
  }

  G4double BetaMin = 1./nMax;

  if ( BetaMin >= 1. ) return StepLimit;

  G4double GammaMin = 1./std::sqrt(1.-BetaMin*BetaMin);

  if (gamma < GammaMin ) return StepLimit;

  G4double kinEmin = mass*(GammaMin-1.);

  G4double RangeMin = G4LossTableManager::Instance()->GetRange(particleType,
                                                               kinEmin,
                                                               couple);
  G4double Range    = G4LossTableManager::Instance()->GetRange(particleType,
                                                               kineticEnergy,
                                                               couple);

  G4double Step = Range - RangeMin;
//  if (Step < 1.*um ) return StepLimit;


  if (Step > 0. && Step < StepLimit) StepLimit = Step; 

  // If user has defined an average maximum number of photons to
  // be generated in a Step, then calculate the Step length for
  // that number of photons. 

  if (fMaxPhotons > 0) {
     // particle charge
     const G4double charge = aParticle->GetDefinition()->GetPDGCharge();

     G4double MeanNumberOfPhotons = 
                      GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

     Step = 0.;
     if (MeanNumberOfPhotons > 0.0) Step = fMaxPhotons / MeanNumberOfPhotons;

     if (Step > 0. && Step < StepLimit) StepLimit = Step;
  }

  // If user has defined an maximum allowed change in beta per step
  if (fMaxBetaChange > 0.) {

     G4double dedx = G4LossTableManager::Instance()->GetDEDX(particleType,
                                                             kineticEnergy,
                                                             couple);

     G4double deltaGamma = gamma - 1./std::sqrt(1.-beta*beta*
                                                (1.-fMaxBetaChange)*
                                                (1.-fMaxBetaChange));

     Step = mass * deltaGamma / dedx;

     if (Step > 0. && Step < StepLimit) StepLimit = Step;

  }

  *condition = StronglyForced;
  return StepLimit;
}

// GetAverageNumberOfPhotons
// -------------------------
// This routine computes the number of Cerenkov photons produced per
// GEANT-unit (millimeter) in the current medium.
//             ^^^^^^^^^^

G4double
  G4Cerenkov_modified::GetAverageNumberOfPhotons(const G4double charge,
                                        const G4double beta, 
                      const G4Material* aMaterial,
                      G4MaterialPropertyVector* Rindex) //const
{

  const G4double Rfact = 369.81/(eV * cm);

#ifdef X_INSTRUMENTED
  std::cout 
       << "G4Cerenkov_modified::GetAverageNumberOfPhotons"
       << " Rfact " << std::fixed << std::setw(10) << std::setprecision(5) << Rfact
       << " eV " << std::fixed << std::setw(10) << std::setprecision(7) << eV
       << " cm " << std::fixed << std::setw(10) << std::setprecision(5) << cm
       << " charge " << std::fixed << std::setw(10) << std::setprecision(5) << charge
       << std::endl
       ;
#endif

  if(beta <= 0.0)return 0.0;

  G4double BetaInverse = 1./beta;

  // Vectors used in computation of Cerenkov Angle Integral:
  //  - Refraction Indices for the current material
  //  - new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
  G4int materialIndex = aMaterial->GetIndex();

  // Retrieve the Cerenkov Angle Integrals for this material  

  G4PhysicsOrderedFreeVector* CerenkovAngleIntegrals =
             (G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(materialIndex));

  if(!(CerenkovAngleIntegrals->IsFilledVectorExist()))return 0.0;

  /*
  // Min and Max photon energies 
  G4double Pmin = Rindex->GetMinLowEdgeEnergy();
  G4double Pmax = Rindex->GetMaxLowEdgeEnergy();

  // Min and Max Refraction Indices 
  G4double nMin = Rindex->GetMinValue();  
  G4double nMax = Rindex->GetMaxValue();

  // Max Cerenkov Angle Integral 
  G4double CAImax = CerenkovAngleIntegrals->GetMaxValue();

  G4double dp, ge;

  // If n(Pmax) < 1/Beta -- no photons generated 

  if (nMax < BetaInverse) {
     dp = 0.0;
     ge = 0.0;
  } 

  // otherwise if n(Pmin) >= 1/Beta -- photons generated  

  else if (nMin > BetaInverse) {
     dp = Pmax - Pmin;  
     ge = CAImax; 
  } 

  // If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
  // we need to find a P such that the value of n(P) == 1/Beta.
  // Interpolation is performed by the GetEnergy() and
  // Value() methods of the G4MaterialPropertiesTable and
  // the GetValue() method of G4PhysicsVector.  

  else {
     Pmin = Rindex->GetEnergy(BetaInverse);
     dp = Pmax - Pmin;

     // need boolean for current implementation of G4PhysicsVector
     // ==> being phased out
     G4bool isOutRange;
     G4double CAImin = CerenkovAngleIntegrals->GetValue(Pmin, isOutRange);
     ge = CAImax - CAImin;

     if (verboseLevel>0) {
        G4cout << "CAImin = " << CAImin << G4endl;
        G4cout << "ge = " << ge << G4endl;
      }
      //////// old version ///////////

  }
 
        */
     ///////// new version ///////////

    G4int    cross_num;
    // G4double cross_up[10];   // max crossing point number : 10
    // G4double cross_down[10];   // max crossing point number : 10
    std::vector<double> the_energies_threshold; // there are several pairs (ranges) at the threshold of 1/beta
    // [ E_l_0, E_r_0, E_l_1, E_r_1, ... ]
    // the energies between [E_l_i, E_r_i] is above the threshold.

     cross_num = 0;
     size_t vec_length = Rindex->GetVectorLength();

     G4double maxRI=(*Rindex)[0]; G4double minRI=(*Rindex)[0];
     for (size_t ii = 1;
             ii < vec_length;
             ++ii) {
         G4double currentRI = (*Rindex)[ii];
         if (currentRI > maxRI) maxRI = currentRI;
         if (currentRI < minRI) minRI = currentRI;
     }



     if (BetaInverse <= minRI) { // All range is OK!

         // cross_up[0] = Rindex->Energy(0);
         // cross_down[0] = Rindex->Energy(vec_length-1);
         cross_num = 1;

         the_energies_threshold.push_back(Rindex->Energy(0));
         the_energies_threshold.push_back(Rindex->Energy(vec_length-1));

         //G4cout << "Range [ " << cross_up[0] << ", " << cross_down[0] << "]" << G4endl;

     } else if (BetaInverse >= maxRI) { // Out of Range 
         cross_num = 0;

     } else {   // between min and max

         // below is impl by Tao Lin
         double currentRI = (*Rindex)[0];
         double currentPM = Rindex->Energy(0);

         // first point
         if (currentRI >= BetaInverse) {
             the_energies_threshold.push_back(currentPM);
         }

         // middle points
         if (vec_length>2) {
             for (size_t ii = 1; ii < vec_length-1; ++ii) {
                 double prevRI = (*Rindex)[ii-1];
                 double prevPM = Rindex->Energy(ii-1);
                 double currentRI = (*Rindex)[ii];
                 double currentPM = Rindex->Energy(ii);

                 // two case here
                 if ( (prevRI >= BetaInverse and currentRI < BetaInverse)
                      or (prevRI < BetaInverse and currentRI >= BetaInverse) ) {
                     double energy_threshold = (BetaInverse-prevRI)/(currentRI-prevRI)*(currentPM-prevPM) + prevPM;
                     the_energies_threshold.push_back(energy_threshold);
                 }
             
             }
         }

         // last point
         currentRI = (*Rindex)[vec_length-1];
         currentPM = Rindex->Energy(vec_length-1);
         if (currentRI >= BetaInverse) {
             the_energies_threshold.push_back(currentPM);
         }

         if ((the_energies_threshold.size()%2) != 0) {
             G4cerr << "ERROR: missing endpoint for the_energies_threshold? "
                    << "The size of the_energies_threshold is "
                    << the_energies_threshold.size()
                    << G4endl;
         }

         cross_num = the_energies_threshold.size() / 2;
         // for (int i = 0; i < cross_num; ++i) {
         //     cross_up[i] = the_energies_threshold[i*2];
         //     cross_down[i] = the_energies_threshold[i*2+1];
         // }

         // END

         // // below is impl by Miao Yu

         // G4double up_leftx[10], up_lefty[10];
         // G4double up_rightx[10], up_righty[10];
         // G4double down_leftx[10], down_lefty[10];
         // G4double down_rightx[10], down_righty[10];
         // G4double extremex[10], extremey[10];
         // G4int num0 = 0;
         // G4int num1 = 0;
         // G4int num2 = 0;

         // double currentRI = (*Rindex)[0];
         // double currentPM = Rindex->Energy(0);
         // if (currentRI == BetaInverse) {
         //     extremex[num2] =  currentPM; 
         //     extremey[num2] = currentRI;
         //     num2++;
         // }

         // for (size_t ii = 1;
         //         ii < vec_length;
         //         ++ii) {
         //     double prevRI = (*Rindex)[ii-1];
         //     double prevPM = Rindex->Energy(ii-1);
         //     double currentRI = (*Rindex)[ii];
         //     double currentPM = Rindex->Energy(ii);

         //     if(prevRI < BetaInverse and currentRI > BetaInverse) { // up
         //         up_leftx[num0] = prevPM;
         //         up_rightx[num0] = currentPM;
         //         up_lefty[num0] = prevRI;
         //         up_righty[num0] = currentRI;
         //         num0++;
         //     } else if(prevRI > BetaInverse and currentRI < BetaInverse) {
         //         down_leftx[num1] = prevPM;
         //         down_rightx[num1] = currentPM;
         //         down_lefty[num1] = prevRI;
         //         down_righty[num1] = currentRI;
         //         num1++;
         //     } else if (currentRI == BetaInverse) {
         //         extremex[num2] = currentPM;
         //         extremey[num2] = currentRI;
         //         num2++;
         //     }
         // }

         // if (num0 - num1 == 1) // up > down
         // {
         //     down_leftx[num1] = Rindex->Energy(vec_length-1);
         //     down_rightx[num1] = down_leftx[num1];
         //     down_lefty[num1] = (*Rindex)[vec_length-1];
         //     down_righty[num1] = down_lefty[num1];
         //     num1++;
         // } else if(num1 - num0 == 1) {
         //     up_leftx[num0] = Rindex->Energy(0);
         //     up_rightx[num0] = up_leftx[num0];
         //     up_lefty[num0] = (*Rindex)[0];
         //     up_righty[num0] = up_lefty[num0];
         //     num0++;
         // }

         // if(num0 != num1) G4cout << "Error: Vector Length Mismatching ! " << G4endl;

         // // linear-interpolation for crossing points:
         // for (int i=0; i<num0; i++) {

         //     if (up_leftx[i] == up_rightx[i]) cross_up[i] = up_leftx[i];
         //     else cross_up[i] = (BetaInverse-up_lefty[i])/(up_righty[i] - up_lefty[i])*(up_rightx[i]-up_leftx[i]) + up_leftx[i];
         //     if (down_leftx[i] == down_rightx[i]) cross_down[i] = down_leftx[i];
         //     else cross_down[i] = (BetaInverse-down_lefty[i])/(down_righty[i] - down_lefty[i])*(down_rightx[i]-down_leftx[i]) + down_leftx[i];
         //     cross_num++;

         //     //G4cout << "Range [ " << cross_up[i] << ", " << cross_down[i] << "]" << G4endl;
         // }
     }
     G4double dp1 = 0; G4double ge1 = 0;
     for (int i=0; i<cross_num; i++) {
        dp1 += the_energies_threshold[2*i+1] - the_energies_threshold[2*i];
        G4bool isOutRange;
        ge1 += CerenkovAngleIntegrals->GetValue(the_energies_threshold[2*i+1], isOutRange) 
               - CerenkovAngleIntegrals->GetValue(the_energies_threshold[2*i], isOutRange);
     }

  // Calculate number of photons 
  //G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
  //                               (dp - ge * BetaInverse*BetaInverse);
  G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
         (dp1 - ge1 * BetaInverse*BetaInverse);


#ifdef X_INSTRUMENTED
  std::cout 
       << "G4Cerenkov_modified::GetAverageNumberOfPhotons"
       << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(5) << BetaInverse
       << " maxRI " << std::fixed << std::setw(10) << std::setprecision(5) << maxRI
       << " minRI " << std::fixed << std::setw(10) << std::setprecision(5) << minRI
       << " cross_num " << cross_num
       << " dp1 " << std::fixed << std::setw(10) << std::setprecision(5) << dp1
       << " dp1/eV " << std::fixed << std::setw(10) << std::setprecision(5) << dp1/eV
       << " ge1 " << std::fixed << std::setw(10) << std::setprecision(5) << ge1
       << " NumPhotons " << std::fixed << std::setw(10) << std::setprecision(5) << NumPhotons
       << std::endl
       ;

  for(int i=0 ; i < cross_num ; i++)
  {

      G4bool isOutRange;
      G4double cai0 = CerenkovAngleIntegrals->GetValue(the_energies_threshold[2*i+0], isOutRange);
      G4double cai1 = CerenkovAngleIntegrals->GetValue(the_energies_threshold[2*i+1], isOutRange);

      std::cout 
           << "G4Cerenkov_modified::GetAverageNumberOfPhotons"
           << " the_energies_threshold[2*i+0]/eV " << std::fixed << std::setw(10) << std::setprecision(5) << the_energies_threshold[2*i+0]/eV
           << " the_energies_threshold[2*i+1]/eV " << std::fixed << std::setw(10) << std::setprecision(5) << the_energies_threshold[2*i+1]/eV
           << " cai0 " << std::fixed << std::setw(20) << std::setprecision(10) << cai0
           << " cai1 " << std::fixed << std::setw(20) << std::setprecision(10) << cai1
           << std::endl 
           ;
  } 
#endif


  return NumPhotons;    
}

void G4Cerenkov_modified::DumpPhysicsTable() const
{
  G4int PhysicsTableSize = thePhysicsTable->entries();
  G4PhysicsOrderedFreeVector *v;

  for (G4int i = 0 ; i < PhysicsTableSize ; i++ ) {
      v = (G4PhysicsOrderedFreeVector*)(*thePhysicsTable)[i];
      v->DumpValues();
  }
}


