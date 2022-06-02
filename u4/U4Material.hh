#pragma once

class G4Material ; 

#include "U4_API_EXPORT.hh"

struct U4_API U4Material
{
    static G4Material* Get(const char* name);
    static G4Material* Get_(const char* name);
    static G4Material* Vacuum(const char* name);
}; 
