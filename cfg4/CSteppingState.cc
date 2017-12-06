#include <sstream>

#include "G4String.hh"
#include "G4VProcess.hh"
#include "G4ProcessVector.hh"

#include "CSteppingState.hh"


std::string CSteppingState::desc() const 
{
    int num_fPostStepGetPhysIntVector = fPostStepGetPhysIntVector ? fPostStepGetPhysIntVector->size() : -1 ; 

    const G4String& procName = fCurrentProcess->GetProcessName() ; 


    std::stringstream ss ; 
    ss << "CSteppingState"
       << " fCurrentProcess " << procName
       << " fPostStepGetPhysIntVector " << num_fPostStepGetPhysIntVector
       << " MAXofPostStepLoops " << MAXofPostStepLoops
       ; 

  

    return ss.str();
}

