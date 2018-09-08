#pragma once

#include "CFG4_API_EXPORT.hh"

#include "G4MaterialPropertyVector.hh"
class G4VParticleChange ; 
class OpticksGenstep ; 

/**
CCerenkovGenerator
===================

Generates photons from gensteps using a VERBATIM copy 
of the G4Cerenkov1042 photon generation loop.

Getting the same photons as a prior run requires:

1. same RINDEX property at the genstep recorded G4Material materialIndex
2. arranging that the same RNG are provided, by controlling the engine  


This is used by CGenstepSource


**/

class CFG4_API CCerenkovGenerator
{
    public:
        static G4MaterialPropertyVector* GetRINDEX(unsigned materialIndex) ;
        static G4VParticleChange* GeneratePhotonsFromGenstep( const OpticksGenstep* gs, unsigned idx )  ;

};



