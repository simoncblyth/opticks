#include <sstream>
#include <cassert>
#include <vector>

#include "NGLM.hpp"
#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "G4ThreeVector.hh"
#include "G4ParticleChange.hh"
#include "G4StepPoint.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4ParticleMomentum.hh"
#include "G4DynamicParticle.hh"
#include "G4OpticalPhoton.hh"
#include "Randomize.hh"

#include "CCerenkovGenerator.hh"
#include "PLOG.hh"



G4MaterialPropertyVector* CCerenkovGenerator::GetRINDEX(unsigned materialIndex) // static
{
    const std::vector<G4Material*>& mtab = *G4Material::GetMaterialTable() ; 

    bool have_material = materialIndex < mtab.size() ; 
    if(!have_material) 
        LOG(fatal) << " missing materialIndex " << materialIndex
                   << " in table of " << mtab.size()
                   ;

    assert( have_material ) ; 
    const G4Material* aMaterial = mtab[materialIndex] ; 

    G4MaterialPropertiesTable* aMaterialPropertiesTable = aMaterial->GetMaterialPropertiesTable(); 
    assert(aMaterialPropertiesTable); 

    G4MaterialPropertyVector* Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX); 
    assert(Rindex);  

    return Rindex ;
}


/**
CCerenkovGenerator::generatePhotonsFromGenstep
-------------------------------------------------

A method of two halves:

1. unpack a genstep item resurrecting the PostStepDoIt context recorded previously 
2. verbatim use of the G4Cerenkov::PostStepDoIt photon generation loop

**/

G4VParticleChange* CCerenkovGenerator::GeneratePhotonsFromGenstep( const OpticksGenstep* gs, unsigned idx ) // static 
{
    unsigned num_gs = gs->getNumGensteps(); 
    bool have_gs = idx < num_gs ; 

    if(!have_gs) 
        LOG(fatal) 
            << " IDX out of range " 
            << " num_gs " << num_gs
            << " idx " << idx
            ;

    assert(have_gs) ; 

    unsigned gencode = gs->getGencode(idx); 
    assert( gencode == CERENKOV ); 


    glm::ivec4 hdr = gs->getHdr(idx); 
    glm::vec4 post = gs->getPositionTime(idx); 
    glm::vec4 dpsl = gs->getDeltaPositionStepLength(idx); 
    glm::vec4 q3   = gs->getQ3(idx); 
    glm::vec4 i3   = gs->getI3(idx); 
    glm::vec4 q4   = gs->getQ4(idx); 
    glm::vec4 q5   = gs->getQ5(idx); 

   // int gencode = hdr.x ;    // enum CERENKOV, SCINTILLATION, TORCH, EMITSOURCE  : but old gensteps just used sign of 1-based index 
    unsigned trackID = hdr.y ;  
    unsigned materialIndex = hdr.z ;  

    //OVERRIDE  : seems to be a problem with the index or order of materials 
    materialIndex = 0 ; 

    unsigned fNumPhotons = hdr.w ; 

    G4ThreeVector x0( post.x, post.y, post.z  ); 
    G4double t0 = post.w*ns ;  

    G4ThreeVector deltaPosition( dpsl.x, dpsl.y, dpsl.z ); 
    G4double stepLength = dpsl.w ;  

    //G4int pdgCode = i3.x ; 
    //G4double pdgCharge = q3.y ;  
    //G4double weight = q3.z ;      // unused is good : means space for the two velocities
    G4double meanVelocity = q3.w ;  

    G4double BetaInverse = q4.x ; 
    G4double Pmin = q4.y ;    // TODO: check units 
    G4double Pmax = q4.z ; 
    //G4double maxCos = q4.w ;

    LOG(info) 
        << " Pmin " << Pmin
        << " Pmax " << Pmax
        << " meanVelocity " << meanVelocity
        ;

    G4double maxSin2 = q5.x ; 
    G4double MeanNumberOfPhotons1 = q5.y ; 
    G4double MeanNumberOfPhotons2 = q5.z ; 
    G4double zero = q5.w ; 
    G4double epsilon = 1e-6 ; 
    assert( std::abs(zero) < epsilon ) ;     // caution with mixed buffers
    // am i storing a int in there, get a very small number ?

    G4double dp = Pmax - Pmin;
    G4ThreeVector p0 = deltaPosition.unit();

    G4ParticleChange* pParticleChange = new G4ParticleChange ;
    G4ParticleChange& aParticleChange = *pParticleChange ; 

    G4Step aStep ;   // dtor will delete points so they must be on heap
    aStep.SetPreStepPoint(new G4StepPoint); 
    aStep.SetPostStepPoint(new G4StepPoint); 
    aStep.SetStepLength(stepLength);  

    G4StepPoint* pPreStepPoint = aStep.GetPreStepPoint();
    G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();

    pPreStepPoint->SetPosition(x0) ; 
    pPostStepPoint->SetPosition(x0+deltaPosition) ; 

    G4Track aTrack ; 
    aTrack.SetTrackID(trackID) ; 

    G4MaterialPropertyVector* Rindex = GetRINDEX(materialIndex) ;  // hmm ordinng problem potential

    G4int verboseLevel = 1 ;   
    using CLHEP::twopi ; 


    /////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  TO CONSIDER :
    // 
    //  Use a C4Cerenkov1041PhotonGenerationLoop.icc to avoid the source duplication 
    //  and use that in two places:
    //   
    //  1. normal PostStepDoIt
    //  2. static G4VParticleChange* C4Cerenkov1042::GenerateSecondaryPhotons( const OpticksGenstep* gs, unsigned idx ) ;  
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  The below is a VERBATIM COPY of the PHOTON GENERATION LOOP from C4Cerenkov1042.cc 
    //  any changes should be marked by ifdef-else preprocessor defines that retain the original 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////


  for (G4int i = 0; i < fNumPhotons; i++) {

      // Determine photon energy

      G4double rand;
      G4double sampledEnergy, sampledRI; 
      G4double cosTheta, sin2Theta;

      // sample an energy

      do {
         rand = G4UniformRand();	
         sampledEnergy = Pmin + rand * dp; 
         sampledRI = Rindex->Value(sampledEnergy);
         cosTheta = BetaInverse / sampledRI;  

         sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
         rand = G4UniformRand();	

        // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
      } while (rand*maxSin2 > sin2Theta);

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


#ifdef HAVE_CHANGED_GENSTEP_TO_STORE_BOTH_VELOCITIES

      G4double deltaTime = delta / (pPreStepPoint->GetVelocity()+
                                      rand*(pPostStepPoint->GetVelocity()-
                                            pPreStepPoint->GetVelocity())*0.5);

#else
      G4double deltaTime = delta / meanVelocity ; 
#endif


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
     G4cout <<"\n Exiting from C4Cerenkov1042::DoIt -- NumberOfSecondaries = "
	    << aParticleChange.GetNumberOfSecondaries() << G4endl;
  }

  return pParticleChange;
}





