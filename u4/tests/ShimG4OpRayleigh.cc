#include "ShimG4OpRayleigh.h"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


ShimG4OpRayleigh::ShimG4OpRayleigh()
    :
    G4OpRayleigh("OpRayleigh",fOptical)
{
} 

ShimG4OpRayleigh::~ShimG4OpRayleigh()
{
}



#ifdef DEBUG_TAG

#include "U4Stack.h"
#include "SEvt.hh"

const bool ShimG4OpRayleigh::FLOAT = getenv("ShimG4OpRayleigh_FLOAT") != nullptr ;
const int  ShimG4OpRayleigh::PIDX  = std::atoi( getenv("PIDX") ? getenv("PIDX") : "-1" ); 

/**
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
-------------------------------------------------------

Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify


**/

// void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
 void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft()
{
    G4double u = G4UniformRand() ; 
    SEvt::AddTag( U4Stack_RayleighDiscreteReset, u ); 

    if(FLOAT)
    {
        float f =  -1.f*std::log( float(u) ) ;  
        theNumberOfInteractionLengthLeft = f ; 
    } 
    else
    {
        theNumberOfInteractionLengthLeft = -1.*G4Log(u) ;  
    }
    theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
}


 G4double ShimG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,  G4double, G4ForceCondition* )
{
     G4double len = G4OpRayleigh::GetMeanFreePath( aTrack, 0, 0 ); 
     //std::cout << "ShimG4OpRayleigh::GetMeanFreePath " << len << std::endl ; 
     return len ; 
}

 G4double ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength(const G4Track& track, G4double   previousStepSize, G4ForceCondition* condition)
{
  if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
    // beggining of tracking (or just after DoIt of this process)
    ResetNumberOfInteractionLengthLeft();
  } else if ( previousStepSize > 0.0) {
    // subtract NumberOfInteractionLengthLeft 
    SubtractNumberOfInteractionLengthLeft(previousStepSize);
  } else {
    // zero step
    //  DO NOTHING
  }
      
  // condition is set to "Not Forced"
  *condition = NotForced;

  // get mean free path
  currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
  
  G4double value;
  if (currentInteractionLength <DBL_MAX) {

     if( FLOAT )
     {
          float fvalue = float(theNumberOfInteractionLengthLeft) * float(currentInteractionLength) ; 
          value = fvalue ; 
     }   
     else
     {
          value = theNumberOfInteractionLengthLeft * currentInteractionLength ; 
     }                

     //std::cout << "ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength " << track.GetTrackID() << std::endl ; 

      if(track.GetTrackID() - 1  == PIDX)
      {
           std::cout 
               << "ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength"
               << " PIDX " << PIDX 
               << " currentInteractionLength " << std::setw(10) << std::fixed << std::setprecision(7) << currentInteractionLength
               << " theNumberOfInteractionLengthLeft " << std::setw(10) << std::fixed << std::setprecision(7) << theNumberOfInteractionLengthLeft
               << " value " << std::setw(10) << std::fixed << std::setprecision(7) << value
               << std::endl
               ;
      } 


  } else {       
    value = DBL_MAX;
  }
#ifdef G4VERBOSE
  if (verboseLevel>1){ 
    G4cout << "G4VDiscreteProcess::PostStepGetPhysicalInteractionLength ";
    G4cout << "[ " << GetProcessName() << "]" <<G4endl;
    track.GetDynamicParticle()->DumpInfo();
    G4cout << " in Material  " <<  track.GetMaterial()->GetName() <<G4endl;
    G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
  }                          
#endif           
  return value;  

}




G4VParticleChange* ShimG4OpRayleigh::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
        aParticleChange.Initialize(aTrack);

        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();

        if (verboseLevel>0) {
                G4cout << "Scattering Photon!" << G4endl;
                G4cout << "Old Momentum Direction: "
                       << aParticle->GetMomentumDirection() << G4endl;
                G4cout << "Old Polarization: "
                       << aParticle->GetPolarization() << G4endl;
        }   

        G4double cosTheta;
        G4ThreeVector OldMomentumDirection, NewMomentumDirection;
        G4ThreeVector OldPolarization, NewPolarization;

        G4double rand, constant;
        G4double CosTheta, SinTheta, SinPhi, CosPhi, unit_x, unit_y, unit_z;

        G4double u ; 
        G4double u_loopexit ; 


        do {
           // Try to simulate the scattered photon momentum direction
           // w.r.t. the initial photon momentum direction

           CosTheta = G4UniformRand();
           SEvt::AddTag( U4Stack_RayleighScatter, CosTheta );   // 0

           SinTheta = std::sqrt(1.-CosTheta*CosTheta);
           // consider for the angle 90-180 degrees

           u = G4UniformRand() ; 
           SEvt::AddTag( U4Stack_RayleighScatter, u );      // 1 

           if (u < 0.5) CosTheta = -CosTheta;

           // simulate the phi angle

           u = G4UniformRand() ; 
           SEvt::AddTag( U4Stack_RayleighScatter, u );    // 2 

           rand = twopi*u;
           SinPhi = std::sin(rand);
           CosPhi = std::cos(rand);

           // start constructing the new momentum direction
       unit_x = SinTheta * CosPhi; 
       unit_y = SinTheta * SinPhi;  
       unit_z = CosTheta; 
       NewMomentumDirection.set (unit_x,unit_y,unit_z);

           // Rotate the new momentum direction into global reference system
           OldMomentumDirection = aParticle->GetMomentumDirection();
           OldMomentumDirection = OldMomentumDirection.unit();
           NewMomentumDirection.rotateUz(OldMomentumDirection);
           NewMomentumDirection = NewMomentumDirection.unit();

           // calculate the new polarization direction
           // The new polarization needs to be in the same plane as the new
           // momentum direction and the old polarization direction
           OldPolarization = aParticle->GetPolarization();
           constant = -NewMomentumDirection.dot(OldPolarization);

           NewPolarization = OldPolarization + constant*NewMomentumDirection;
           NewPolarization = NewPolarization.unit();

           // There is a corner case, where the Newmomentum direction
           // is the same as oldpolariztion direction:
           // random generate the azimuthal angle w.r.t. Newmomentum direction

           u = G4UniformRand() ; 
           SEvt::AddTag( U4Stack_RayleighScatter, u );   // 3

           if (NewPolarization.mag() == 0.) {
              rand = u*twopi;
              NewPolarization.set(std::cos(rand),std::sin(rand),0.);
              NewPolarization.rotateUz(NewMomentumDirection);
           } else {
              // There are two directions which are perpendicular
              // to the new momentum direction
              if (u < 0.5) NewPolarization = -NewPolarization;
           }

       // simulate according to the distribution cos^2(theta)
           cosTheta = NewPolarization.dot(OldPolarization);

           u_loopexit = G4UniformRand() ;
           SEvt::AddTag( U4Stack_RayleighScatter, u_loopexit );   // 4

          // Loop checking, 13-Aug-2015, Peter Gumplinger
        } while (std::pow(cosTheta,2) < u_loopexit );

        aParticleChange.ProposePolarization(NewPolarization);
        aParticleChange.ProposeMomentumDirection(NewMomentumDirection);

        if (verboseLevel>0) {
                G4cout << "New Polarization: "
                     << NewPolarization << G4endl;
                G4cout << "Polarization Change: "
                     << *(aParticleChange.GetPolarization()) << G4endl;
                G4cout << "New Momentum Direction: "
                     << NewMomentumDirection << G4endl;
                G4cout << "Momentum Change: "
                     << *(aParticleChange.GetMomentumDirection()) << G4endl;
        }

        return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
}













#endif



