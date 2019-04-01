#pragma once

#include <map>
#include <string>

// okc-
class OpticksHub ; 
class OpticksQuery ; 

// npy-
class NCSG ; 
class NGeoTestConfig ; 

// ggeo-
class GGeoTest ; 
class GMaterial ;
class GCSG ; 

// cfg4-
class CMaker ; 
class CPropLib ; 
class CSensitiveDetector ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;


#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"

/**

CTestDetector
=================

*CTestDetector* is a :doc:`CDetector` subclass that
constructs simple Geant4 detector test geometries based on commandline specifications
parsed and represented by an instance of :doc:`../npy/NGeoTestConfig`.

Canonical instance resides in CGeometry and is instanciated by CGeometry::init
when --test option is used. After the instanciation the CDetector::attachSurfaces
is invoked.

**/


class CFG4_API CTestDetector : public CDetector
{
 public:
    CTestDetector(OpticksHub* hub, OpticksQuery* query=NULL, CSensitiveDetector* sd=NULL);
  private:
    void init();

  private:
    G4VPhysicalVolume* makeDetector();
    G4VPhysicalVolume* makeDetector_NCSG();
    G4VPhysicalVolume* makeChildVolume(const NCSG* csg, const char* lvn, const char* pvn, G4LogicalVolume* mother);
    G4VPhysicalVolume* makeVolumeUniverse(const NCSG* csg);

  private:
    GGeoTest*          m_geotest ; 
    NGeoTestConfig*    m_config ; 
    CMaker*            m_maker ; 

};



