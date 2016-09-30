#pragma once


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

Canonical instance lives in CGenerator

**/


class CFG4_API CTorchSource: public CSource
{
  public:
    CTorchSource(TorchStepNPY* torch, unsigned int verbosity);
  private:
    void init();
    void configure();
  public:
    virtual ~CTorchSource();
    void GeneratePrimaryVertex(G4Event *evt);
    void SetVerbosity(int vL);  // override
  private:
    TorchStepNPY*         m_torch ;
    G4SPSPosDistribution* m_posGen;
    G4SPSAngDistribution* m_angGen;
    G4SPSEneDistribution* m_eneGen;
    G4SPSRandomGenerator* m_ranGen;
    NPY<float>*           m_primary ; 

};


