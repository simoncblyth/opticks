// op --cgdmldetector
#pragma once

// ggeo-
class GCache ; 

// optickscore-
class OpticksQuery ;

#include "CDetector.hh"
class CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(GCache* cache, OpticksQuery* query);
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    void addMPT();
};

inline CGDMLDetector::CGDMLDetector(GCache* cache, OpticksQuery* query)
  : 
  CDetector(cache, query)
{
    init();
}

inline CGDMLDetector::~CGDMLDetector()
{
}


