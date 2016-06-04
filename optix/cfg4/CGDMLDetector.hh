// op --cgdmldetector
#pragma once


// optickscore-
class Opticks ;
class OpticksQuery ;

#include "CDetector.hh"
class CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(Opticks* cache, OpticksQuery* query);
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    void addMPT();
};

inline CGDMLDetector::CGDMLDetector(Opticks* cache, OpticksQuery* query)
  : 
  CDetector(cache, query)
{
    init();
}

inline CGDMLDetector::~CGDMLDetector()
{
}


