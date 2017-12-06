#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

//#include "G4SteppingManager.hh"
#include "G4StepStatus.hh"

class G4VProcess ;
class G4ProcessVector ; 

struct CFG4_API CSteppingState 
{
    G4VProcess*      fCurrentProcess ; 
    G4ProcessVector* fPostStepGetPhysIntVector ; 
    //G4SelectedPostStepDoItVector*  fSelectedPostStepDoItVector ;
    size_t MAXofPostStepLoops ; 

    G4StepStatus     fStepStatus ; 


    std::string desc() const ; 
};

#include "CFG4_TAIL.hh"


