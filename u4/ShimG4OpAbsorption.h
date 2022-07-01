#pragma once
/**
ShimG4OpAbsorption
====================

G4OpAbsorption : public G4VDiscreteProcess

**/

#include "G4OpAbsorption.hh"
#include <csignal>


class ShimG4OpAbsorption : public G4OpAbsorption 
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
/**
ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft
--------------------------------------------------------

Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify

**/

const bool ShimG4OpAbsorption::FLOAT = getenv("ShimG4OpAbsorption_FLOAT") != nullptr ;

//inline void ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
inline void ShimG4OpAbsorption::ResetNumberOfInteractionLengthLeft()
{
    //std::cout << "ShimG4OpAbsorption::FLOAT " << FLOAT << std::endl ; 
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

inline G4double ShimG4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,  G4double, G4ForceCondition* )
{
     G4double len = G4OpAbsorption::GetMeanFreePath( aTrack, 0, 0 ); 
     //std::cout << "ShimG4OpAbsorption::GetMeanFreePath " << len << std::endl ; 

     //std::raise(SIGINT); 

     return len ; 
}


inline G4double ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength(const G4Track& track, G4double   previousStepSize, G4ForceCondition* condition)
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
  if (currentInteractionLength <DBL_MAX) 
  {
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
 
