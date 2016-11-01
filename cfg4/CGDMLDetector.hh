// op --cgdmldetector
#pragma once


class OpticksQuery ; // okc-
class OpticksHub ;   // okg-

#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"

/**

CGDMLDetector
~~~~~~~~~~~~~~

*CGDMLDetector* is a :doc:`CDetector` subclass that
loads Geant4 GDML persisted geometry files.

**/

class CFG4_API CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(OpticksHub* hub, OpticksQuery* query);
    void saveBuffers();
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    void addMPT();
};


