#pragma once

struct STranche ; 
class NPho ; 
template <typename T> class NPY ; 
class GenstepNPY ;



class G4PrimaryVertex ;

#include <string>
#include "plog/Severity.h"
#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

/**
CInputPhotonSource
====================

Canonical m_source instance lives in CGenerator, created by CGenerator::initInputPhotonSource

Implements the G4VPrimaryGenerator interface : GeneratePrimaryVertex



**/


class CFG4_API CInputPhotonSource: public CSource
{
        static const plog::Severity LEVEL ; 
    public:
        CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, GenstepNPY* gsnpy );
        void reset(); 
    public:
        virtual ~CInputPhotonSource();
    public:
        // G4VPrimaryGenerator interface
        void     GeneratePrimaryVertex(G4Event *evt);
    public:
        unsigned  getNumG4Event() const ; 
        unsigned  getNumPhotonsPerG4Event() const;
        std::string desc() const ;
    private:
        G4PrimaryVertex*      convertPhoton(unsigned pho_index);
    private:
        bool                  m_sourcedbg ; 
        NPho*                 m_pho ;
        GenstepNPY*           m_gsnpy ; 
    private:
        unsigned              m_numPhotonsPerG4Event ;
        unsigned              m_numPhotons ;
        STranche*             m_tranche ; 
        unsigned              m_gpv_count ;   // count calls to GeneratePrimaryVertex

};


