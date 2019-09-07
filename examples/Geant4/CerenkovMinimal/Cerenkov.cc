/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "G4Cerenkov.hh"
#include "Cerenkov.hh"
#include "G4LossTableManager.hh"

using CLHEP::um ;


Cerenkov::Cerenkov( const G4String& processName, G4ProcessType type)
    :
    G4Cerenkov(processName, type)
{
    G4cout << "Cerenkov::Cerenkov" << G4endl ; 
} 

G4bool Cerenkov::IsApplicable(const G4ParticleDefinition& aParticleType)
{
    G4bool a = G4Cerenkov::IsApplicable(aParticleType);
    G4cout 
        << "Cerenkov::IsApplicable"
        << std::setw(30) << aParticleType.GetParticleName() 
        << " : "
        << a 
        << G4endl 
        ;
    return a ;      
}

void Cerenkov::BuildPhysicsTable(const G4ParticleDefinition& aParticleType)
{
    G4Cerenkov::BuildPhysicsTable(aParticleType);
    G4cout 
        << "Cerenkov::BuildPhysicsTable"
        << std::setw(30) << aParticleType.GetParticleName() 
        << G4endl
        ;

    //std::cout << *thePhysicsTable << std::endl ; 
    DumpPhysicsTable(); 
}


G4VParticleChange* Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    G4cout << "Cerenkov::PostStepDoIt"
           << G4endl 
            ; 
    return G4Cerenkov::PostStepDoIt(aTrack, aStep);  
}





/*
G4double Cerenkov::PostStepGetPhysicalInteractionLength(
                                       const G4Track& aTrack,
                                       G4double ignored,
                                       G4ForceCondition* condition)
{
    const G4Material* aMaterial = aTrack.GetMaterial();
    G4int materialIndex = aMaterial->GetIndex();

    typedef G4PhysicsOrderedFreeVector FV ; 
    FV* CerenkovAngleIntegrals_0 = (FV*)((*thePhysicsTable)(materialIndex));
    FV* CerenkovAngleIntegrals_1 = (FV*)((*thePhysicsTable)[materialIndex]);
    assert( CerenkovAngleIntegrals_0 == CerenkovAngleIntegrals_1 ); 

    G4cout << "CAI:" << CerenkovAngleIntegrals_0 << " mtIdx " << materialIndex << " " << aMaterial->GetName()   ; 

    G4double psgpil = G4Cerenkov::PostStepGetPhysicalInteractionLength(aTrack, ignored, condition);

    bool is_DBL_MAX = psgpil == DBL_MAX ; 

    G4cout << "psgpil:" << psgpil << " is_DBL_MAX " << is_DBL_MAX ; 

    return psgpil ;   
}
*/




G4double Cerenkov::PostStepGetPhysicalInteractionLength(
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
                     Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");

        G4double nMax;
        if (Rindex) {
           nMax = Rindex->GetMaxValue();
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

           Step = 0.;
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

           Step = mass * deltaGamma / dedx;

           if (Step > 0. && Step < StepLimit) StepLimit = Step;

        }

        *condition = StronglyForced;
        return StepLimit;
}
