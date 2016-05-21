// op --cgdmldetector
#pragma once

// ggeo-
class GCache ;

#include "CDetector.hh"
class CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(GCache* cache);
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    void addMPT();
};

inline CGDMLDetector::CGDMLDetector(GCache* cache)
  : 
  CDetector(cache)
{
    init();
}

inline CGDMLDetector::~CGDMLDetector()
{
}


