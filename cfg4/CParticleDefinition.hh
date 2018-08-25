#pragma once

class G4ParticleDefinition ;

#include "CFG4_API_EXPORT.hh"


class CFG4_API CParticleDefinition 
{
    public:
        static G4ParticleDefinition* Find(const char* name) ; 
 
};


