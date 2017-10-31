#pragma once

template <typename T> class NPY ; 
class GenstepNPY ;

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
  private:
    void init();
    void configure();
  public:
    virtual ~CInputPhotonSource();
    void GeneratePrimaryVertex(G4Event *evt);

  private:
    bool                  m_sourcedbg ; 
    NPY<float>*           m_input_photons ;
    GenstepNPY*           m_gsnpy ; 
    NPY<float>*           m_primary ; 

};


