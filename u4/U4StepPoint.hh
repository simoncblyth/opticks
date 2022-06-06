#pragma once

struct sphoton ; 
class G4StepPoint ; 

#include "U4_API_EXPORT.hh"

struct U4_API U4StepPoint
{
    static void Update(sphoton& photon, const G4StepPoint* point);
}; 


