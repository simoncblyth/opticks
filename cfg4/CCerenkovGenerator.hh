#pragma once

#include "CFG4_API_EXPORT.hh"

class G4VParticleChange ; 

class NGS ; 
template <typename T> class NPY ; 

/**
CCerenkovGenerator
================

**/

class CFG4_API CCerenkovGenerator
{
    public:
        CCerenkovGenerator(NPY<float>* gs);
    public:
        G4VParticleChange* generatePhotonsFromGenstep( unsigned i );
    private:
        NGS*                  m_gs ;


};



