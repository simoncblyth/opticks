// op --cgdmldetector
#pragma once


class OpticksQuery ; // okc-
class OpticksHub ;   // okg-
class G4VPhysicalVolume ; 

#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"
#include "plog/Severity.h"


/**

CGDMLDetector
==============

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
    G4VPhysicalVolume* parseGDML(const char* path) const ;

    void addMPT();
    void addSurfaces();
    void kludge_cathode_efficiency();

    plog::Severity m_level ;

};


