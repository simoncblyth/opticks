#pragma once

#include "G4Threading.hh"

class NGunConfig ; 

#include "CSource.hh"

class CGunSource: public CSource
{
  public:
    CGunSource(NGunConfig* config, unsigned int verbosity);
  private:
    void init();
    void configure();
    void SetVerbosity(int vL);
  public:
    virtual ~CGunSource();
    void GeneratePrimaryVertex(G4Event *evt);
  private:
    NGunConfig*           m_config ; 
    G4int                 m_verbosityLevel;
    G4Mutex               m_mutex;
};


inline CGunSource::CGunSource(NGunConfig* config, unsigned int verbosity)  
    :
    CSource(),
    m_config(config),
    m_verbosityLevel(verbosity)
{
    init();
}





