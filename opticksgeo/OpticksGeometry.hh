#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>

class Opticks ; 
class OpticksAttrSeq ;
template <typename> class OpticksCfg ;
class GGeo ; 
class GMesh ;
class GMergedMesh ;

#include "OKGEO_API_EXPORT.hh"
class OKGEO_API OpticksGeometry {
   public:
       OpticksGeometry(Opticks* opticks);
  public:
       void loadGeometry();
  public:
       GGeo*           getGGeo();
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
       Opticks*             m_opticks ; 
       OpticksCfg<Opticks>* m_fcfg ;
       GGeo*                m_ggeo ; 
       GMergedMesh*         m_mesh0 ;  

};


