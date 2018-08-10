#include "G4Cerenkov.hh"
#include "Cerenkov.hh"
#include "PLOG.hh"

Cerenkov::Cerenkov( const G4String& processName, G4ProcessType type)
    :
    G4Cerenkov(processName, type)
{
    LOG(info) << "." ; 
} 

G4bool Cerenkov::IsApplicable(const G4ParticleDefinition& aParticleType)
{
    G4bool a = G4Cerenkov::IsApplicable(aParticleType);
    LOG(info) 
        << std::setw(30) << aParticleType.GetParticleName() 
        << " : "
        << a 
        ;
    return a ;      
}

void Cerenkov::BuildPhysicsTable(const G4ParticleDefinition& aParticleType)
{
    G4Cerenkov::BuildPhysicsTable(aParticleType);
    LOG(info) 
        << std::setw(30) << aParticleType.GetParticleName() 
        ;

    //std::cout << *thePhysicsTable << std::endl ; 
    DumpPhysicsTable(); 
}

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

    LOG(info) << "CAI:" << CerenkovAngleIntegrals_0 << " mtIdx " << materialIndex << " " << aMaterial->GetName()   ; 

    G4double psgpil = G4Cerenkov::PostStepGetPhysicalInteractionLength(aTrack, ignored, condition);

    bool is_DBL_MAX = psgpil == DBL_MAX ; 

    LOG(info) << "psgpil:" << psgpil << " is_DBL_MAX " << is_DBL_MAX ; 

    return psgpil ;   
}


G4VParticleChange* Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
    LOG(info) << "." ; 
    return G4Cerenkov::PostStepDoIt(aTrack, aStep);  
}


