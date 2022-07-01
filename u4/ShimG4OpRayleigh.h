#pragma once
/**
ShimG4OpRayleigh 
===================

class G4OpRayleigh : public G4VDiscreteProcess

**/

#include "G4OpRayleigh.hh"
class ShimG4OpRayleigh : public G4OpRayleigh 
{
    public:
#ifdef DEBUG_TAG
        static const bool FLOAT ; 
        void ResetNumberOfInteractionLengthLeft(); 
        G4double GetMeanFreePath(const G4Track& aTrack,
                 G4double ,
                 G4ForceCondition* );

      G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& track,
                 G4double   previousStepSize,
                 G4ForceCondition* condition
                );  

#endif
}; 


#ifdef DEBUG_TAG
const bool ShimG4OpRayleigh::FLOAT = getenv("ShimG4OpRayleigh_FLOAT") != nullptr ;

/**
ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft
-------------------------------------------------------

Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify


**/

//inline void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
inline void ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft()
{
    G4double u = G4UniformRand() ; 
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


inline G4double ShimG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,  G4double, G4ForceCondition* )
{
     G4double len = G4OpRayleigh::GetMeanFreePath( aTrack, 0, 0 ); 
     //std::cout << "ShimG4OpRayleigh::GetMeanFreePath " << len << std::endl ; 
     return len ; 
}

inline G4double ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength(const G4Track& track, G4double   previousStepSize, G4ForceCondition* condition)
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

#endif



