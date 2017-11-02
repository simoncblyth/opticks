#pragma once

#include <map>
#include <string>

// okc-
class OpticksHub ; 
class OpticksQuery ; 

// ggeo-
class GGeoTest ; 
class GGeoTestConfig ; 
class GMaterial ;
class GCSG ; 

// cfg4-
class CMaker ; 
class CPropLib ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;

//
// Hmm for fully controlled cfg4- testing need 
// to construct composite boolean G4 geometries for things like PMTs
//  
// Maybe GParts specs not appropriate level to start from for that ? 
// Need a higher level ? detdesc ?
//
// Parsing detdesc is a shortterm DYB specific kludge, the analytic geometry info 
// should eventually live withing the G4DAE exported file (GDML style). 
// Given this, the approach for making G4 PMTs can be a kludge too. 
// Does not need to be very general, probably will only every use for PMTs,
// as this approach too much effort for full geometries.
//  
// See:  
//       pmt-ecd
//           python detdesc parsing into parts buffer
//
//       GMaker::makeZSphereIntersect    
//           converts high level params (two sphere radii and z offsets) of convex lens
//           into parts  
//
//  Possible approach:
//       during the pmt- python partitioning to create the parts buffer 
//       write a sidecar buffer of high level params, that can be used for G4 boolean CSG
//       


#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"


/**

CTestDetector
=================

*CTestDetector* is a :doc:`CDetector` subclass that
constructs simple Geant4 detector test geometries based on commandline specifications
parsed and represented by an instance of :doc:`../ggeo/GGeoTestConfig`.

Canonical instance resides in CGeometry and is instanciated by CGeometry::init
when --test option is used. After the instanciation the CDetector::attachSurfaces
is invoked.



**/


class CFG4_API CTestDetector : public CDetector
{
 public:
    CTestDetector(OpticksHub* hub, GGeoTest* geotest, OpticksQuery* query=NULL);
  private:
    void init();

  private:
    G4VPhysicalVolume* makeDetector();
    G4VPhysicalVolume* makeDetector_NCSG();

  private:
    G4VPhysicalVolume* makeDetector_OLD();
    void               makePMT_OLD(G4LogicalVolume* mother);
    void               kludgePhotoCathode_OLD();
    G4LogicalVolume*   makeLV_OLD(GCSG* csg, unsigned int i);

  private:
    GGeoTest*          m_geotest ; 
    GGeoTestConfig*    m_config ; 
    CMaker*            m_maker ; 

};



