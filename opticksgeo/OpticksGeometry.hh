#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>

class OpticksHub ; 
class Opticks ; 
class Composition ; 
class OpticksAttrSeq ;
template <typename> class OpticksCfg ;
class GGeo ; 
class GMesh ;
class GMergedMesh ;

/**
OpticksGeometry
================

Canonical m_geometry instance resides in okg/OpticksHub 
and is instanciated by OpticksHub::init which 
happens within the ok/OKMgr or okg4/OKG4Mgr ctors.


**/


#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksGeometry {
   public:
       OpticksGeometry(OpticksHub* hub);
  public:
       void loadGeometry();
  public:
       GGeo*           getGGeo();

       unsigned        getTarget();
       void            setTarget(unsigned target=0, bool aim=true);
       unsigned        getTargetDeferred();

       glm::vec4       getCenterExtent();
       OpticksAttrSeq* getMaterialNames();
       OpticksAttrSeq* getBoundaryNames();
       std::map<unsigned int, std::string> getBoundaryNamesMap();
  private: 
       void loadGeometryBase();
       void modifyGeometry();
       void fixGeometry();
       void registerGeometry();
       void configureGeometry(); 
   private:
       void init();
   private:
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       Composition*         m_composition ; 
       OpticksCfg<Opticks>* m_fcfg ;
       GGeo*                m_ggeo ; 
       GMergedMesh*         m_mesh0 ;  
       unsigned             m_target ;
       unsigned             m_target_deferred ;
       unsigned             m_verbosity ;
     

};


