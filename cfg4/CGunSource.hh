#pragma once


class NGunConfig ; 

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

class CFG4_API CGunSource: public CSource
{
  public:
    CGunSource(int verbosity);
    virtual ~CGunSource();
    void configure(NGunConfig* gc);
  private:
    void init();
    void SetVerbosity(int vL);
  public:
    void GeneratePrimaryVertex(G4Event *evt);
  private:
    NGunConfig*           m_config ; 

};


