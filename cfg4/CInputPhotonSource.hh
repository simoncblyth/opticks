#pragma once

struct STranche ; 
class NPho ; 
template <typename T> class NPY ; 
class GenstepNPY ;



class G4PrimaryVertex ;

#include <string>
#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

/**
CInputPhotonSource
====================

Canonical instance lives in CGenerator

**/


class CFG4_API CInputPhotonSource: public CSource
{
    public:
        CInputPhotonSource(Opticks* ok, NPY<float>* input_photons, GenstepNPY* gsnpy, unsigned int verbosity);
    public:
        virtual ~CInputPhotonSource();
        void     GeneratePrimaryVertex(G4Event *evt);
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
        NPY<float>*           m_primary ; 
        unsigned              m_gpv_count ;   // count calls to GeneratePrimaryVertex

        //bool                  m_mask ;  // --mask 
        //unsigned              m_mask_skip ;
        //unsigned              m_mask_take ;
};


