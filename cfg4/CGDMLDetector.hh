// op --cgdmldetector
#pragma once


class OpticksQuery ; // okc-
class OpticksHub ;   // okg-
class G4VPhysicalVolume ; 


class CSensitiveDetector ; 

#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"
#include "plog/Severity.h"

/**
CGDMLDetector
==============

*CGDMLDetector* is a :doc:`CDetector` subclass that
loads Geant4 GDML persisted geometry files, 
from m_ok->getGDMLPath().

**/

class CFG4_API CGDMLDetector : public CDetector
{
  public:
    CGDMLDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd);
    void saveBuffers();
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    G4VPhysicalVolume* parseGDML(const char* path) const ;

    void sortMaterials();
    void addMPTLegacyGDML();
    void standardizeGeant4MaterialProperties(); // by adoption of those from Opticks  


    //void addSD();    <-- too early SD only gets created later at CG4 
    void addSurfaces();
    //void kludge_cathode_efficiency();

    plog::Severity m_level ;

};


