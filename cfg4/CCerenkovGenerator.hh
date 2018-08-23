#pragma once

#include "CFG4_API_EXPORT.hh"
#include "G4MaterialPropertyVector.hh"

class G4VParticleChange ; 
class NGS ; 

/**
CCerenkovGenerator
================

**/

class CFG4_API CCerenkovGenerator
{
    public:
        CCerenkovGenerator(NGS* gs);
    public:
        G4MaterialPropertyVector* getRINDEX(unsigned materialIndex);
        G4VParticleChange* generatePhotonsFromGenstep( unsigned i );
    private:
        NGS*                  m_gs ;


};



