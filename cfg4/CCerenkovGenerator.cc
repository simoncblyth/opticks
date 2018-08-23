#include <cassert>

#include "NGLM.hpp"
#include "NGS.hpp"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"



#include "CCerenkovGenerator.hh"
#include "PLOG.hh"


CCerenkovGenerator::CCerenkovGenerator(NPY<float>* gs)  
    :
    m_gs(new NGS(gs)),
{
}

G4VParticleChange* CCerenkovGenerator::generatePhotonsFromGenstep( unsigned i )
{
    glm::ivec4 hdr = m_gs->getHdr(i); 
    glm::vec4 post = m_gs->getPositionTime(i); 
    glm::vec4 dpsl = m_gs->getDeltaPositionStepLength(i); 

    glm::vec4 q3   = m_gs->getQ3(i); 
    glm::vec4 i3   = m_gs->getI3(i); 
    glm::vec4 q4   = m_gs->getQ4(i); 
    glm::vec4 q5   = m_gs->getQ5(i); 

    unsigned id = hdr.x ; 
    unsigned trackID = hdr.y ;  
    unsigned materialIndex = hdr.z ;  
    unsigned fNumPhotons = hdr.w ; 

    G4ThreeVector x0( post.x, post.y, post.z  ); 
    G4double t0 = post.w ;  

    G4ThreeVector deltaPosition( dpsl.x, dpsl.y, dpsl.z ); 
    G4double stepLength = dpsl.w ;  

    G4int pdgCode = i3.x ; 
    G4double pdgCharge = q3.y ;  
    G4double weight = q3.z ;  
    G4double meanVelocity = q3.w ;  

    G4double BetaInverse = q4.x ; 
    G4double Pmin = q4.y ; 
    G4double Pmax = q4.z ; 
    G4double maxCos = q4.w ;

    G4double maxSin2 = q5.x ; 
    G4double MeanNumberOfPhotons1 = q5.y ; 
    G4double MeanNumberOfPhotons2 = q5.z ; 
    G4double zero = q5.w ; 
    assert( zero == 0. ) ; 


    G4double dp = Pmax - Pmin;
    G4ThreeVector p0 = deltaPosition().unit();

    // resurrect PostStepDoIt context 

    G4ParticleChange* pParticleChange = new G4ParticleChange ;
    G4ParticleChange& aParticleChange = *pParticleChange ; 

    G4StepPoint preStepPoint ; 
    G4StepPoint postStepPoint ;

    preStepPoint.SetPosition(x0) ; 
    postStepPoint.SetPosition(x0+deltaPosition) ; 

    G4Step aStep ;
    aStep.SetPreStepPoint(&preStepPoint); 
    aStep.SetPostStepPoint(&postStepPoint); 
    aStep.SetStepLength( stepLength );  

    G4Track aTrack ; 
    aTrack.SetTrackID(trackId) ; 

    G4int verboseLevel = 1 ;   


    const std::vector<G4Material*>& mtab = *G4Material::GetMaterialTable() ; 
    assert( materialIndex < mtab.size() ) ; 
    const G4Material* aMaterial = mtab[materialIndex] ; 

    G4MaterialPropertiesTable* aMaterialPropertiesTable = aMaterial->GetMaterialPropertiesTable(); 
    assert(aMaterialPropertiesTable); 

    G4MaterialPropertyVector* Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX); 
    assert(Rindex);  

    using CLHEP::twopi ; 

    //////////////////////////////////////////////////////////////////////////////////////
    ///// below is VERBATIM PHOTON GENERATION LOOP  from  C4Cerenkov1042.cc 
    ////   keep changes to a minimum , and mark them all 
    //////////////////////////////////////////////////////////////////////////////////////


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

#ifdef HAVE_MOCKED_A_TOUCHABLE
      aSecondaryTrack->SetTouchableHandle(
                               aStep.GetPreStepPoint()->GetTouchableHandle());
#endif

      aSecondaryTrack->SetParentID(aTrack.GetTrackID());

      aParticleChange.AddSecondary(aSecondaryTrack);
  }

  if (verboseLevel>0) {
     G4cout <<"\n Exiting from C4Cerenkov1042::DoIt -- NumberOfSecondaries = "
	    << aParticleChange.GetNumberOfSecondaries() << G4endl;
  }

  return pParticleChange;



}

