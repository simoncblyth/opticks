#pragma once

// TODO: avoid duplication of this between here and optixrap-/photon.h
//       via a high level opticks pkg for containing high level 
//       common things such as there enums
//
//       remember that this gets parsed to get the strings, so that should move too

enum
{
    CERENKOV          = 0x1 <<  0,    
    SCINTILLATION     = 0x1 <<  1,        
    MISS              = 0x1 <<  2,  
    BULK_ABSORB       = 0x1 <<  3,  
    BULK_REEMIT       = 0x1 <<  4,  
    BULK_SCATTER      = 0x1 <<  5,  
    SURFACE_DETECT    = 0x1 <<  6,  
    SURFACE_ABSORB    = 0x1 <<  7,  
    SURFACE_DREFLECT  = 0x1 <<  8,  
    SURFACE_SREFLECT  = 0x1 <<  9,  
    BOUNDARY_REFLECT  = 0x1 << 10, 
    BOUNDARY_TRANSMIT = 0x1 << 11, 
    TORCH             = 0x1 << 12, 
    NAN_ABORT         = 0x1 << 13
}; // processes


#include "G4StepStatus.hh"
#include "G4OpBoundaryProcess.hh"
#include <string>

std::string OpStepString(const G4StepStatus status);

std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status);

std::string OpFlagString(const unsigned int flag);
std::string OpFlagSequenceString(const unsigned long long seqhis);


