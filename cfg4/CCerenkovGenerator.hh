#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"
#include "G4MaterialPropertyVector.hh"

class G4VParticleChange ; 
class NGS ; 
class CPhotonCollector ; 

/**
CCerenkovGenerator
================

**/

class CFG4_API CCerenkovGenerator
{
    public:
        CCerenkovGenerator(NGS* gs);
    public:
        G4MaterialPropertyVector* getRINDEX(unsigned materialIndex) const ;
        G4VParticleChange* generatePhotonsFromGenstep( unsigned idx ) const ;
        void collectSecondaryPhotons( const G4VParticleChange* pc, unsigned idx ) ;
        void generateAndCollectPhotonsFromGenstep( unsigned idx ) ;
        void savePhotons(const char* path) const ; 
        std::string desc() const ;   
    private:
        NGS*                  m_gs ;
        CPhotonCollector*     m_photon_collector ; 


};



