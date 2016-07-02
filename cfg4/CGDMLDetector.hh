// op --cgdmldetector
#pragma once


// okc-
class Opticks ;
class OpticksQuery ;

#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"

class CFG4_API CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(Opticks* cache, OpticksQuery* query);
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    void addMPT();
};


