#pragma once

#include <string>
template <typename T> class NPY ; 
class TorchStepNPY ; 

class G4SPSPosDistribution ;
class G4SPSAngDistribution ;
class G4SPSEneDistribution ;
class G4SPSRandomGenerator ;

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"


/**
CTorchSource
=============

Canonical m_source instance lives in CGenerator
and is instanciated by CGenerator::initSource

**/


class CFG4_API CTorchSource: public CSource
{
    public:
        CTorchSource(Opticks* ok, TorchStepNPY* torch, unsigned int verbosity);
    private:
        void init();
        void configure();
    public:
        virtual ~CTorchSource();
        void setVerbosity(int verbosity) ;
        void GeneratePrimaryVertex(G4Event *evt);
        std::string desc() const ;
    private:
        TorchStepNPY*         m_torch ;
        bool                  m_torchdbg ; 
        int                   m_verbosity ; 
        unsigned              m_num_photons_total ; 
        unsigned              m_num_photons_per_g4event ; 
        unsigned              m_num_photons ; 

        G4SPSPosDistribution* m_posGen;
        G4SPSAngDistribution* m_angGen;
        G4SPSEneDistribution* m_eneGen;
        G4SPSRandomGenerator* m_ranGen;

        NPY<float>*           m_primary ; 

};


