#pragma once

#include "CFG4_API_EXPORT.hh"
#include <string>

/**
C4PhotonCollector
===================

**/

class G4VParticleChange ; 
class CPhotonCollector ; 
template <typename T> class NPY ;

class CFG4_API C4PhotonCollector 
{
    public:
        C4PhotonCollector(); 
    public:
        NPY<float>*  getPhoton() const ;
    public:
        void collectSecondaryPhotons( const G4VParticleChange* pc, unsigned idx ) ;
        void savePhotons(const char* path) const ; 
        std::string desc() const ; 
    private:
        CPhotonCollector*     m_photon_collector ; 


};

