#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>

class OpticksHub ; 
class Opticks ; 
class Composition ; 
//class OpticksAttrSeq ;
template <typename> class OpticksCfg ;
class GGeo ; 
class GMesh ;
class GMergedMesh ;

/**
OpticksGeometry
================

Actually OpticksGGeo would be a better name, this acts as a higher 
level holder of GGeo with a triangulated (G4DAE) focus.  
Anything related to analytic (GLTF) should not live here, OpticksHub 
would be more appropriate.

Canonical m_geometry instance resides in okg/OpticksHub 
and is instanciated by OpticksHub::init which 
happens within the ok/OKMgr or okg4/OKG4Mgr ctors.


Dev History
-------------

* started as spillout from monolithic GGeo



**/


#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksGeometry {
   public:
       OpticksGeometry(OpticksHub* hub);
  public:
       void loadGeometry();
  public:
       GGeo*           getGGeo();

       // move up to hub  
       //OpticksAttrSeq* getMaterialNames();
       //OpticksAttrSeq* getBoundaryNames();
       //std::map<unsigned int, std::string> getBoundaryNamesMap();

  private: 
       void loadGeometryBase();
       void fixGeometry();
   private:
       void init();
   private:
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       Composition*         m_composition ; 
       OpticksCfg<Opticks>* m_fcfg ;
       GGeo*                m_ggeo ; 
       unsigned             m_verbosity ;
     

};


