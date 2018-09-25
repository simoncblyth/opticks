#include <sstream>
#include <cassert>
#include <vector>

#include "NGLM.hpp"
#include "Opticks.hh"
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
#include "GBndLib.hh"
#include "GLMFormat.hpp"
#include "PLOG.hh"

#define ALIGN_DEBUG 1
#ifdef ALIGN_DEBUG
#include "CAlignEngine.hh"
#endif


/**
CCerenkovGenerator::GetRINDEX
-------------------------------

TODO : materialline (texture line) saved in the genstep for GPU tex lookups, needs
to be translated back into an ordinary Geant4 material index.


**/

G4MaterialPropertyVector* CCerenkovGenerator::GetRINDEX(unsigned materialIndex) // static
{
    const std::vector<G4Material*>& mtab = *G4Material::GetMaterialTable() ; 

    unsigned num_material = mtab.size() ; 

    bool have_material = materialIndex < num_material ; 
    if(!have_material) 
        LOG(fatal) << " missing materialIndex " << materialIndex
                   << " in table of " << num_material
                   ;

    assert( have_material ) ; 
    const G4Material* aMaterial = mtab[materialIndex] ; 

    G4MaterialPropertiesTable* aMaterialPropertiesTable = aMaterial->GetMaterialPropertiesTable(); 
    assert(aMaterialPropertiesTable); 

    G4MaterialPropertyVector* Rindex = aMaterialPropertiesTable->GetProperty(kRINDEX); 
    assert(Rindex);  

    G4MaterialPropertyVector* Rindex2 = aMaterialPropertiesTable->GetProperty("RINDEX"); 
    assert(Rindex2);  


    LOG(error) 
         << " aMaterial " << (void*)aMaterial
         << " aMaterial.Name " << aMaterial->GetName()
         << " materialIndex " << materialIndex
         << " num_material " << num_material
         << " Rindex " << Rindex 
         << " Rindex2 " << Rindex2 
         ;


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
    unsigned materialLine = hdr.z ;  
    unsigned materialIndex = GBndLib::MaterialIndexFromLine( materialLine ) ; 
    // back translate a texture material line into a material index

    int fNumPhotons = hdr.w ; 

    G4ThreeVector x0( post.x, post.y, post.z  ); 
    G4double t0 = post.w*ns ;  

    LOG(info) 
        << " genstep_idx " << idx
        << " num_gs " << num_gs
        << " materialLine " << materialLine
        << " materialIndex " << materialIndex 
        << gpresent("post", post ) 
        ;

    G4ThreeVector deltaPosition( dpsl.x, dpsl.y, dpsl.z ); 
    G4double stepLength = dpsl.w ;  

    //G4int pdgCode = i3.x ; 
    //G4double pdgCharge = q3.y ;  
    //G4double weight = q3.z ;      // unused is good : means space for the two velocities
    G4double preVelocity = q3.w ;  

    G4double BetaInverse = q4.x ; 
    G4double Pmin = q4.y ;    
    G4double Pmax = q4.z ; 

    G4double wavelength_min = h_Planck*c_light/Pmax ;
    G4double wavelength_max = h_Planck*c_light/Pmin ;

    //G4double maxCos = q4.w ;

    G4double maxSin2 = q5.x ; 
    G4double MeanNumberOfPhotons1 = q5.y ; 
    G4double MeanNumberOfPhotons2 = q5.z ; 
    G4double postVelocity = q5.w ; 

    //G4double dp = Pmax - Pmin;   // <-- precision loss if energies travel as MeV  : TODO get energies to travel in eV
    G4ThreeVector p0 = deltaPosition.unit();

    LOG(info) 
        << " Pmin " << Pmin
        << " Pmax " << Pmax
        << " wavelength_min(nm) " << wavelength_min/nm
        << " wavelength_max(nm) " << wavelength_max/nm
        << " preVelocity " << preVelocity
        << " postVelocity " << postVelocity
        ;


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

    pPreStepPoint->SetVelocity(preVelocity) ; 
    pPostStepPoint->SetVelocity(postVelocity) ; 


    G4Track aTrack ; 
    aTrack.SetTrackID(trackID) ; 

    G4MaterialPropertyVector* Rindex = GetRINDEX(materialIndex) ; 

    G4double Pmin2 = Rindex->GetMinLowEdgeEnergy();
    G4double Pmax2 = Rindex->GetMaxLowEdgeEnergy();
    G4double dp2 = Pmax2 - Pmin2;

    G4double epsilon = 1e-6 ; 
    bool Pmin_match = std::abs( Pmin2 - Pmin ) < epsilon ; 
    bool Pmax_match = std::abs( Pmax2 - Pmax ) < epsilon ; 
   
    if(!Pmin_match || !Pmax_match)
        LOG(fatal) 
            << " Pmin " << Pmin
            << " Pmin2 (MinLowEdgeEnergy) " << Pmin2
            << " dif " << std::abs( Pmin2 - Pmin )
            << " epsilon " << epsilon
            << " Pmin(nm) " << h_Planck*c_light/Pmin/nm
            << " Pmin2(nm) " << h_Planck*c_light/Pmin2/nm
            ;

    if(!Pmax_match || !Pmin_match)
        LOG(fatal) 
            << " Pmax " << Pmax
            << " Pmax2 (MaxLowEdgeEnergy) " << Pmax2
            << " dif " << std::abs( Pmax2 - Pmax )
            << " epsilon " << epsilon
            << " Pmax(nm) " << h_Planck*c_light/Pmax/nm
            << " Pmax2(nm) " << h_Planck*c_light/Pmax2/nm
            ;

    bool with_key = Opticks::HasKey() ; 
    if(with_key)
    {
        assert( Pmin_match && "material mismatches genstep source material" ); 
        assert( Pmax_match && "material mismatches genstep source material" ); 
    }
    else
    {
        LOG(warning) << "permissive generation for legacy gensteps " ;
    }

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
    //  When by mistake a material like vacuum which has no-Cerenkov is used : this will go into a tailspin 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  The below is a VERBATIM COPY of the PHOTON GENERATION LOOP from C4Cerenkov1042.cc 
    //  any changes should be marked by ifdef-else preprocessor defines that retain the original 
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ALIGN_DEBUG
    int pindex = 0 ; 
    LOG(error) 
        << " genstep_idx " << idx
        << " fNumPhotons " << fNumPhotons
        << " pindex " << pindex
        ;
#endif



  for (G4int i = 0; i < fNumPhotons; i++) {

      // Determine photon energy
#ifdef ALIGN_DEBUG
      CAlignEngine::SetSequenceIndex(i) ; 
      G4double rand2 ;
#endif    

      G4double rand;
      G4double sampledEnergy, sampledRI; 
      G4double cosTheta, sin2Theta;

      // sample an energy

      do {
         rand = G4UniformRand();	
         //sampledEnergy = Pmin + rand * dp; 
         sampledEnergy = Pmin2 + rand * dp2 ; 
         sampledRI = Rindex->Value(sampledEnergy);
         cosTheta = BetaInverse / sampledRI;  

#ifdef ALIGN_DEBUG
         G4double sampledWavelength = h_Planck*c_light/sampledEnergy ;

         if( i == pindex ) LOG(verbose)
                          << " gcp.u0 " << rand
                          << " sampledEnergy " << sampledEnergy
                          << " sampledWavelength " << sampledWavelength/nm
                          << " hc/nm " << h_Planck*c_light/nm 
                          << " sampledRI " << sampledRI
                          ;   
#endif    

         sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
         rand = G4UniformRand();	

#ifdef ALIGN_DEBUG
         if( i == pindex ) LOG(verbose) << "gcp.u1 " << rand ;   
#endif    

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


#ifdef ALIGN_DEBUG
      if( i == pindex ) 
          LOG(error) 
              << "gcp.u2 " << rand 
              << " dir (" 
              << " " << photonMomentum.x()
              << " " << photonMomentum.y()
              << " " << photonMomentum.z()
              << " )"
              << " pol (" 
              << " " << photonPolarization.x()
              << " " << photonPolarization.y()
              << " " << photonPolarization.z()
              << " )"
              ;   
#endif    



      // Generate new G4Track object:

      G4double NumberOfPhotons, N;

      do {
         rand = G4UniformRand();

#ifdef ALIGN_DEBUG
         if( i == pindex ) LOG(error) << "gcp.u3 " << rand ;   
#endif    

         NumberOfPhotons = MeanNumberOfPhotons1 - rand *
                                (MeanNumberOfPhotons1-MeanNumberOfPhotons2);


#ifdef ALIGN_DEBUG
         rand2 = G4UniformRand();
         N = rand2 *
                        std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);

         if( i == pindex ) LOG(error) << "gcp.u4 " << rand2 ;   
#else

         N = G4UniformRand() *
                        std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);

#endif

        // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
      } while (N > NumberOfPhotons);

      G4double delta = rand * aStep.GetStepLength();


//#ifdef HAVE_CHANGED_GENSTEP_TO_STORE_BOTH_VELOCITIES

      G4double deltaTime = delta / (pPreStepPoint->GetVelocity()+
                                      rand*(pPostStepPoint->GetVelocity()-
                                            pPreStepPoint->GetVelocity())*0.5);

//#else
//      G4double deltaTime = delta / meanVelocity ; 
//#endif


      G4double aSecondaryTime = t0 + deltaTime;

      G4ThreeVector aSecondaryPosition = x0 + rand * aStep.GetDeltaPosition();

#ifdef ALIGN_DEBUG
      if( i == pindex ) 
          LOG(error) 
              << "gcp.post ("
              << " " << std::fixed <<  aSecondaryPosition.x() 
              << " " << std::fixed <<  aSecondaryPosition.y() 
              << " " << std::fixed <<  aSecondaryPosition.z()
              << " : " << std::fixed <<  aSecondaryTime
              << ")"
              ;
#endif

      G4Track* aSecondaryTrack = 
               new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);

      aSecondaryTrack->SetTouchableHandle(
                               aStep.GetPreStepPoint()->GetTouchableHandle());

      aSecondaryTrack->SetParentID(aTrack.GetTrackID());

      aParticleChange.AddSecondary(aSecondaryTrack);

#ifdef ALIGN_DEBUG
      CAlignEngine::SetSequenceIndex(-1) ; 
#endif    

  }

  if (verboseLevel>0) {
     G4cout <<"\n Exiting from C4Cerenkov1042::DoIt -- NumberOfSecondaries = "
	    << aParticleChange.GetNumberOfSecondaries() << G4endl;
  }

  return pParticleChange;
}





