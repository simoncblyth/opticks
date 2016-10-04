/**
 * \class DsG4Cerenkov
 *
 * \brief A slightly modified version of G4Cerenkov
 *
 * It is modified to take a given weight to use to reduce the number
 * of opticalphotons that are produced.  They can then later be
 * up-weighted.
 *
 * The modification adds the weight, its accessors and adds
 * implementation to AlongStepDoIt().  We must copy-and-modify instead
 * of inherit because certain needed data members are private and so
 * we can not just override AlongStepDoIt() in our own subclass.
 *
 * This was taken from G4.9.1p1
 *
 * bv@bnl.gov Mon Feb  4 15:52:16 2008
 * Initial mod to support weighted opticalphotons.
 * The mods to dywCerenkov by Jianglai 09-06-2006 were used for guidance.
 *
 * Jul. 27, 2009  wangzhe
 *     ApplyWaterQe: apply all available QE when optical photons are created.
 *     This should be used with WaterCerenQeApplied of DsPmtSensDet.
 *     All modification are enclosed by "wangzhe" and "wz" for
 *     begin and end respectively.
 *     m_qeScale, etc. were copied to here from DsPmtSensDet.
 */

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
// $Id: G4Cerenkov.cc,v 1.26 2008/11/14 20:16:51 gum Exp $
// GEANT4 tag $Name: geant4-09-02 $
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
#include "G4Poisson.hh"
#include "G4EmProcessSubType.hh"

#include "G4LossTableManager.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleDefinition.hh"
#include "G4Version.hh"

#include "DsG4Cerenkov.h"

#include "DsPhotonTrackInfo.h"
#include "DsG4CompositeTrackInfo.h"
using namespace std;


using CLHEP::twopi ; 
using CLHEP::um ; 
using CLHEP::cm ; 
using CLHEP::eV ; 
using CLHEP::eplus ; 


/////////////////////////
// Class Implementation  
/////////////////////////

        //////////////
        // Operators
        //////////////

// G4Cerenkov::operator=(const G4Cerenkov &right)
// {
// }

        /////////////////
        // Constructors
        /////////////////

DsG4Cerenkov::DsG4Cerenkov(const G4String& processName, G4ProcessType type)
           : G4VProcess(processName, type)
           , fApplyPreQE(false)
           , fPreQE(1.)
{

       /*
        G4cout << "DsG4Cerenkov::DsG4Cerenkov constructor" << G4endl;
        G4cout << "NOTE: this is now a G4VProcess!" << G4endl;
        G4cout << "Required change in UserPhysicsList: " << G4endl;
        G4cout << "change: pmanager->AddContinuousProcess(theCerenkovProcess);" << G4endl; // 
        G4cout << "to:     pmanager->AddProcess(theCerenkovProcess);" << G4endl;
        G4cout << "        pmanager->SetProcessOrdering(theCerenkovProcess,idxPostStep);" << G4endl;
        */

        SetProcessSubType(fCerenkov);

	fTrackSecondariesFirst = false;
	fMaxBetaChange = 0.;
	fMaxPhotons = 0;
        fPhotonWeight = 1.0;    // Daya Bay mod, bv@bnl.gov

        thePhysicsTable = NULL;

	if (verboseLevel>0) {
           G4cout << GetProcessName() << " is created " << G4endl;
	}

	BuildThePhysicsTable();
	
	// wangzhe
	fApplyWaterQe = false;
	m_qeScale = 1.0/0.9;
	// wz
}

// G4Cerenkov::G4Cerenkov(const G4Cerenkov &right)
// {
// }

        ////////////////
        // Destructors
        ////////////////

DsG4Cerenkov::~DsG4Cerenkov() 
{
	if (thePhysicsTable != NULL) {
	   thePhysicsTable->clearAndDestroy();
           delete thePhysicsTable;
	}
}

        ////////////
        // Methods
        ////////////

// PostStepDoIt
// -------------
//
G4VParticleChange*
DsG4Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)

// This routine is called for each tracking Step of a charged particle
// in a radiator. A Poisson-distributed number of photons is generated
// according to the Cerenkov formula, distributed evenly along the track
// segment and uniformly azimuth w.r.t. the particle direction. The 
// parameters are then transformed into the Master Reference System, and 
// they are added to the particle change. 

{
	//////////////////////////////////////////////////////
	// Should we ensure that the material is dispersive?
	//////////////////////////////////////////////////////

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

	const G4MaterialPropertyVector* Rindex = 
                aMaterialPropertiesTable->GetProperty("RINDEX"); 
        if (!Rindex) return pParticleChange;

        // particle charge
        const G4double charge = aParticle->GetDefinition()->GetPDGCharge();

        // particle beta
        const G4double beta = (pPreStepPoint ->GetBeta() +
                               pPostStepPoint->GetBeta())/2.;

	G4double MeanNumberOfPhotons = 
                 GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

        if (MeanNumberOfPhotons <= 0.0) {

                // return unchanged particle and no secondaries

                aParticleChange.SetNumberOfSecondaries(0);
 
                return pParticleChange;

        }

        G4double step_length;
        step_length = aStep.GetStepLength();

	MeanNumberOfPhotons = MeanNumberOfPhotons * step_length;
	G4bool ApplyWaterQE = fApplyWaterQe && aMaterial->GetName().contains("Water");

	// Reduce generated photons by given photon weight
	// Daya Bay mod, bv@bnl.gov
	if (verboseLevel>0) {
	  G4cout << "DsG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons 
		 << " before dividing by fPhotonWeight " << fPhotonWeight << G4endl;
	}
	MeanNumberOfPhotons/=fPhotonWeight;
	if (verboseLevel>0) {
	  G4cout << "DsG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons 
		 << " before multiplying by fPreQE " << fPreQE 
		 << " (only if fApplyPreQE=" << fApplyPreQE << " is set true " << G4endl;
	}
	if ( fApplyPreQE ) {
	  // if WaterQE is applied, it's corrected by the fPreQE.
	  MeanNumberOfPhotons *= fPreQE;
	}
	G4int NumPhotons = (G4int) G4Poisson(MeanNumberOfPhotons);
	if (verboseLevel>0) {
	  G4cout << "DsG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons
		 << " as mean of poission used to calculate NumPhotons " << NumPhotons
		 << G4endl;
	}
	if (NumPhotons <= 0) {
		// return unchanged particle and no secondaries  
		aParticleChange.SetNumberOfSecondaries(0);
                return pParticleChange;
	}

	////////////////////////////////////////////////////////////////

	aParticleChange.SetNumberOfSecondaries(NumPhotons);

        if (fTrackSecondariesFirst) {
           if (aTrack.GetTrackStatus() == fAlive )
                   aParticleChange.ProposeTrackStatus(fSuspend);
        }
	
	////////////////////////////////////////////////////////////////

#if ( G4VERSION_NUMBER > 1000 )
	G4double Pmin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMinLowEdgeEnergy();
	G4double Pmax = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMaxLowEdgeEnergy();  // 1-bin different ?
	G4double nMax = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMaxValue();
#else
	G4double Pmin = Rindex->GetMinPhotonEnergy();
	G4double Pmax = Rindex->GetMaxPhotonEnergy();
	G4double nMax = Rindex->GetMaxProperty();
#endif
	G4double dp = Pmax - Pmin;


        G4double BetaInverse = 1./beta;

	G4double maxCos = BetaInverse / nMax; 
	G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);

        const G4double beta1 = pPreStepPoint ->GetBeta();
        const G4double beta2 = pPostStepPoint->GetBeta();

        G4double MeanNumberOfPhotons1 =
                     GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
        G4double MeanNumberOfPhotons2 =
                     GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);

	for (G4int i = 0; i < NumPhotons; i++) {
	  // Determine photon energy
	  G4double rand=0;
	  G4double sampledEnergy=0, sampledRI=0; 
	  G4double cosTheta=0, sin2Theta=0;
	  
	  // sample an energy
	  do {
	    rand = G4UniformRand();	
	    sampledEnergy = Pmin + rand * dp; 
#if ( G4VERSION_NUMBER > 1000 )
	    sampledRI = Rindex->Value(sampledEnergy);
#else
	    sampledRI = Rindex->GetProperty(sampledEnergy);
#endif
	    cosTheta = BetaInverse / sampledRI;  
	    
	    sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
	    rand = G4UniformRand();	
	    
	  } while (rand*maxSin2 > sin2Theta);
	  
	  // wangzhe
	  // kill a optical photon according to the QE(energy) probability function
	  G4double qe=1.;
	  if ( ApplyWaterQE ) {
	    G4double uni;
	    // 0.6: For now, hard code "extra" decrease in efficiency for water shield PMTs to match G4dyb.
	    // m_qeScale: 1.0/0.9
	    G4double effqe=qe=0.6*m_qeScale*GetPoolPmtQe(sampledEnergy);
	    if ( fApplyPreQE ) {
	      // take into account preapplied maximal QE 
	      effqe/=fPreQE;
	      if ( effqe>1. ) G4cerr<<"WaterPMT efficiency>1. This means that used CerenPhotonScaleWeight is too big."<<G4endl;
	    }
	    uni = G4UniformRand();
	    //G4cout <<"qe= "<<qe<<" uni= "<<uni<<" energy= "<<sampledEnergy/CLHEP::eV<<" eV, "
	    //	     <<"raw QE= "<<GetPoolPmtQe(sampledEnergy)<<G4endl;
	    
	    if ( uni >= effqe ) {
	      continue;
	    }
	  }
	  // wz
	  
	  // Generate random position of photon on cone surface 
	  // defined by Theta 
	  rand = G4UniformRand();
	  
	  G4double phi = twopi*rand;
	  G4double sinPhi = sin(phi);
	  G4double cosPhi = cos(phi);
	  
	  // calculate x,y, and z components of photon energy
	  // (in coord system with primary particle direction 
	  //  aligned with the z axis)
	  
	  G4double sinTheta = sqrt(sin2Theta); 
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
	    new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(), 
				  photonMomentum);
	  aCerenkovPhoton->SetPolarization
	    (photonPolarization.x(),
	     photonPolarization.y(),
	     photonPolarization.z());
	  
	  aCerenkovPhoton->SetKineticEnergy(sampledEnergy);
	  
	  // Generate new G4Track object:
	  
	  G4double delta, NumberOfPhotons, N;
	  
	  do {
	    rand = G4UniformRand();
	    delta = rand * aStep.GetStepLength();
	    NumberOfPhotons = MeanNumberOfPhotons1 - delta *
	      (MeanNumberOfPhotons1-MeanNumberOfPhotons2)/
	      aStep.GetStepLength();
	    N = G4UniformRand() *
	      std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);
	  } while (N > NumberOfPhotons);
	  
	  G4double deltaTime = delta /
	    ((pPreStepPoint->GetVelocity()+
	      pPostStepPoint->GetVelocity())/2.);
	  
	  G4double aSecondaryTime = t0 + deltaTime;
	  
	  G4ThreeVector aSecondaryPosition =
	    x0 + rand * aStep.GetDeltaPosition();
	  
	  G4Track* aSecondaryTrack = 
	    new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);
	  
	  // set user track info
	  DsG4CompositeTrackInfo* comp=new DsG4CompositeTrackInfo();
	  DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
	  if ( ApplyWaterQE ) {
	    trackinf->SetMode(DsPhotonTrackInfo::kQEWater);
	    trackinf->SetQE(qe);
	  }
	  else if ( fApplyPreQE ) {
	    trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
	    trackinf->SetQE(fPreQE);
	  }
	  comp->SetPhotonTrackInfo(trackinf);
	  aSecondaryTrack->SetUserInformation(comp);
	  
	  aSecondaryTrack->SetTouchableHandle(
					      aStep.GetPreStepPoint()->GetTouchableHandle());
	  
	  aSecondaryTrack->SetParentID(aTrack.GetTrackID());
	  
	  aParticleChange.AddSecondary(aSecondaryTrack);
	  
	  // Daya Bay mods, bv@bnl.gov
	  aSecondaryTrack->SetWeight(fPhotonWeight*aTrack.GetWeight());
	  aParticleChange.SetSecondaryWeightByProcess(true);
	  if (verboseLevel>0) {
	    G4cout << "DsG4Cerenkov  aSecondaryTrack->SetWeight( fPhotonWeight="<<fPhotonWeight<<" * aTrack.GetWeight()= " << aTrack.GetWeight() 
		   << ") aSecondaryTrack->GetWeight() " << aSecondaryTrack->GetWeight() << G4endl;
	  }
	}
	
	if (verboseLevel>0) {
	G4cout << "\n Exiting from DsG4Cerenkov::DoIt -- NumberOfSecondaries = " 
	     << aParticleChange.GetNumberOfSecondaries() << G4endl;
	}

        return pParticleChange;
}

// BuildThePhysicsTable for the Cerenkov process
// ---------------------------------------------
//

void DsG4Cerenkov::BuildThePhysicsTable()
{
	if (thePhysicsTable) return;

	const G4MaterialTable* theMaterialTable=
	 		       G4Material::GetMaterialTable();
	G4int numOfMaterials = G4Material::GetNumberOfMaterials();

	// create new physics table
	
	thePhysicsTable = new G4PhysicsTable(numOfMaterials);

	// loop for materials

        //G4cerr << "Building physics table with " << numOfMaterials << " materials" << G4endl;

	for (G4int i=0 ; i < numOfMaterials; i++)
	{
		G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector =
					new G4PhysicsOrderedFreeVector();

		// Retrieve vector of refraction indices for the material
		// from the material's optical properties table 

		G4Material* aMaterial = (*theMaterialTable)[i];

		G4MaterialPropertiesTable* aMaterialPropertiesTable =
				aMaterial->GetMaterialPropertiesTable();

		if (aMaterialPropertiesTable) {

		   G4MaterialPropertyVector* theRefractionIndexVector = 
		    	   aMaterialPropertiesTable->GetProperty("RINDEX");

		   if (theRefractionIndexVector) {
		
		      // Retrieve the first refraction index in vector
		      // of (photon energy, refraction index) pairs 

#if ( G4VERSION_NUMBER > 1000 )
		      G4double currentRI = (*theRefractionIndexVector)[0] ;
#else
		      theRefractionIndexVector->ResetIterator();
		      ++(*theRefractionIndexVector);	// advance to 1st entry 

		      G4double currentRI = theRefractionIndexVector->
		  			   GetProperty();

#endif
		      if (currentRI > 1.0) {

			 // Create first (photon energy, Cerenkov Integral)
			 // pair  

			 G4double currentCAI = 0.0;
#if ( G4VERSION_NUMBER > 1000 )
			 G4double currentPM = theRefractionIndexVector->Energy(0);
#else
			 G4double currentPM = theRefractionIndexVector->
			 			 GetPhotonEnergy();
#endif
			 aPhysicsOrderedFreeVector->
			 	 InsertValues(currentPM , currentCAI);



			 // Set previous values to current ones prior to loop

			 G4double prevPM  = currentPM;
			 G4double prevCAI = currentCAI;
                	 G4double prevRI  = currentRI;

			 // loop over all (photon energy, refraction index)
			 // pairs stored for this material  

#if ( G4VERSION_NUMBER > 1000 )
             for (size_t ii = 1; ii < theRefractionIndexVector->GetVectorLength(); ++ii)
             {
				currentRI=(*theRefractionIndexVector)[ii];
                currentPM = theRefractionIndexVector->Energy(ii);
#else
			 while(++(*theRefractionIndexVector))
			 {
				currentRI=theRefractionIndexVector->	
						GetProperty();
				currentPM = theRefractionIndexVector->
						GetPhotonEnergy();
#endif

				currentCAI = 0.5*(1.0/(prevRI*prevRI) +
					          1.0/(currentRI*currentRI));

				currentCAI = prevCAI + 
					     (currentPM - prevPM) * currentCAI;

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

G4double DsG4Cerenkov::GetMeanFreePath(const G4Track&,
                                           G4double,
                                           G4ForceCondition*)
{
        return 1.;
}

G4double DsG4Cerenkov::PostStepGetPhysicalInteractionLength(
                                           const G4Track& aTrack,
                                           G4double,
                                           G4ForceCondition* condition)
{
        *condition = NotForced;
        G4double StepLimit = DBL_MAX;

        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
        const G4Material* aMaterial = aTrack.GetMaterial();
        const G4MaterialCutsCouple* couple = aTrack.GetMaterialCutsCouple();

        const G4double kineticEnergy = aParticle->GetKineticEnergy();
        const G4ParticleDefinition* particleType = aParticle->GetDefinition();
        const G4double mass = particleType->GetPDGMass();

        // particle beta
        const G4double beta = aParticle->GetTotalMomentum() /
                              aParticle->GetTotalEnergy();
        // particle gamma
        const G4double gamma = 1./std::sqrt(1.-beta*beta);

        G4MaterialPropertiesTable* aMaterialPropertiesTable =
                            aMaterial->GetMaterialPropertiesTable();

        const G4MaterialPropertyVector* Rindex = NULL;

        if (aMaterialPropertiesTable)
                     Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");

        G4double nMax;
        if (Rindex) {
#if ( G4VERSION_NUMBER > 1000 )
           nMax = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMaxValue();
#else
           nMax = Rindex->GetMaxProperty();
#endif
        } else {
           return StepLimit;
        }

        G4double BetaMin = 1./nMax;
        if ( BetaMin >= 1. ) return StepLimit;

        G4double GammaMin = 1./std::sqrt(1.-BetaMin*BetaMin);

        if (gamma < GammaMin ) return StepLimit;

        G4double kinEmin = mass*(GammaMin-1.);

        G4double RangeMin = G4LossTableManager::Instance()->
                                                   GetRange(particleType,
                                                            kinEmin,
                                                            couple);
        G4double Range    = G4LossTableManager::Instance()->
                                                   GetRange(particleType,
                                                            kineticEnergy,
                                                            couple);

        G4double Step = Range - RangeMin;
        if (Step < 1.*um ) return StepLimit;

        if (Step > 0. && Step < StepLimit) StepLimit = Step; 

        // If user has defined an average maximum number of photons to
        // be generated in a Step, then calculate the Step length for
        // that number of photons. 
 
        if (fMaxPhotons > 0) {

           // particle charge
           const G4double charge = aParticle->
                                   GetDefinition()->GetPDGCharge();

	   G4double MeanNumberOfPhotons = 
                    GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

           G4double Step = 0.;
           if (MeanNumberOfPhotons > 0.0) Step = fMaxPhotons /
                                                 MeanNumberOfPhotons;

           if (Step > 0. && Step < StepLimit) StepLimit = Step;
        }

        // If user has defined an maximum allowed change in beta per step
        if (fMaxBetaChange > 0.) {

           G4double dedx = G4LossTableManager::Instance()->
                                                   GetDEDX(particleType,
                                                           kineticEnergy,
                                                           couple);

           G4double deltaGamma = gamma - 
                                 1./std::sqrt(1.-beta*beta*
                                                 (1.-fMaxBetaChange)*
                                                 (1.-fMaxBetaChange));

           G4double Step = mass * deltaGamma / dedx;

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
DsG4Cerenkov::GetAverageNumberOfPhotons(const G4double charge,
                              const G4double beta, 
			      const G4Material* aMaterial,
			      const G4MaterialPropertyVector* Rindex) const
{
	const G4double Rfact = 369.81/(eV * cm);

        if(beta <= 0.0)return 0.0;

        G4double BetaInverse = 1./beta;

	// Vectors used in computation of Cerenkov Angle Integral:
	// 	- Refraction Indices for the current material
	//	- new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
        //G4cerr << "In Material getting index: " << aMaterial->GetName() << G4endl;
	G4int materialIndex = aMaterial->GetIndex();
        //G4cerr << "\tindex=" << materialIndex << G4endl;

	// Retrieve the Cerenkov Angle Integrals for this material  
    // G4PhysicsVector* pv = (*thePhysicsTable)(materialIndex);
	G4PhysicsOrderedFreeVector* CerenkovAngleIntegrals =
	(G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(materialIndex));

        if(!(CerenkovAngleIntegrals->IsFilledVectorExist()))return 0.0;

	// Min and Max photon energies 
#if ( G4VERSION_NUMBER > 1000 )
    // NB poorly named methods, actually back/front see G4PhysicsOrderedFreeVector.icc
	G4double Pmin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMinLowEdgeEnergy();
	G4double Pmax = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMaxLowEdgeEnergy();

	// Min and Max Refraction Indices 
	G4double nMin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMinValue();	
	G4double nMax = const_cast<G4MaterialPropertyVector*>(Rindex)->GetMaxValue();
#else
	G4double Pmin = Rindex->GetMinPhotonEnergy();
	G4double Pmax = Rindex->GetMaxPhotonEnergy();

	// Min and Max Refraction Indices 
	G4double nMin = Rindex->GetMinProperty();	
	G4double nMax = Rindex->GetMaxProperty();
#endif

	// Max Cerenkov Angle Integral 
	G4double CAImax = CerenkovAngleIntegrals->GetMaxValue();

	G4double dp=0, ge=0;

	// If n(Pmax) < 1/Beta -- no photons generated 

	if (nMax < BetaInverse) {
		dp = 0;
		ge = 0;
	} 

	// otherwise if n(Pmin) >= 1/Beta -- photons generated  

	else if (nMin > BetaInverse) {
		dp = Pmax - Pmin;	
		ge = CAImax; 
	} 

	// If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
	// we need to find a P such that the value of n(P) == 1/Beta.
	// Interpolation is performed by the GetPhotonEnergy() and
	// GetProperty() methods of the G4MaterialPropertiesTable and
	// the GetValue() method of G4PhysicsVector.  

	else {
#if ( G4VERSION_NUMBER > 1000 )
		Pmin = const_cast<G4MaterialPropertyVector*>(Rindex)->GetEnergy(BetaInverse);
#else
		Pmin = Rindex->GetPhotonEnergy(BetaInverse);
#endif
		dp = Pmax - Pmin;

		// need boolean for current implementation of G4PhysicsVector
		// ==> being phased out
		G4bool isOutRange;
		G4double CAImin = CerenkovAngleIntegrals->
                                  GetValue(Pmin, isOutRange);
		ge = CAImax - CAImin;

		if (verboseLevel>0) {
			G4cout << "CAImin = " << CAImin << G4endl;
			G4cout << "ge = " << ge << G4endl;
		}
	}
	
	// Calculate number of photons 
	G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
                                 (dp - ge * BetaInverse*BetaInverse);

	return NumPhotons;		
}


// wangzhe
// get the raw pmt photocathode QE
G4double DsG4Cerenkov::GetPoolPmtQe(G4double energy) const
{
  static bool first = true;
  static G4Material* bialkali = 0;
  if(first) {
    bialkali = G4Material::GetMaterial("/dd/Materials/Bialkali");
    if( bialkali ==0 ) {
      G4cout<<"Error: DsG4Cerenkov::Can't find material bialkali."<<G4endl;
    }
    first = false;
  }
  
  G4MaterialPropertyVector* qevec = bialkali->GetMaterialPropertiesTable()->GetProperty("EFFICIENCY");
#if ( G4VERSION_NUMBER > 1000 )
  return qevec->Value(energy);
#else
  return qevec->GetProperty(energy);
#endif

}
// wz
