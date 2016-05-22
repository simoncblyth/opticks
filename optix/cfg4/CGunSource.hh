#pragma once


class NGunConfig ; 

#include "CSource.hh"

class CGunSource: public CSource
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


inline CGunSource::CGunSource(int verbosity)  
    :
    CSource(verbosity),
    m_config(NULL)
{
    init();
}





