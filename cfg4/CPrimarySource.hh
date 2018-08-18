#pragma once

#include "CSource.hh"
#include "CFG4_API_EXPORT.hh"

class G4Event ; 
class G4PrimaryVertex ;


/**
CPrimarySource
================

**/

class CFG4_API CPrimarySource: public CSource
{
  public:
    CPrimarySource(Opticks* ok, int verbosity);
    virtual ~CPrimarySource();
  private:
    void init();
  public:
    void GeneratePrimaryVertex(G4Event *evt);
  public:
  private:

};


