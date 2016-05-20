#pragma once

#include "G4Threading.hh"

#include "CSource.hh"

class CGunSource: public CSource
{
  public:
    CGunSource(unsigned int verbosity);
  private:
    void init();
    void configure();
    void SetVerbosity(int vL);
  public:
    virtual ~CGunSource();
    void GeneratePrimaryVertex(G4Event *evt);
  private:
    G4int                 m_verbosityLevel;
    G4Mutex               m_mutex;
};


inline CGunSource::CGunSource(unsigned int verbosity)  
    :
    CSource(),
    m_verbosityLevel(verbosity)
{
    init();
}





