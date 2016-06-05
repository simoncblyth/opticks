#pragma once

#include <cstddef>
#include <string>
#include <map>

#include <glm/glm.hpp>

class Opticks ; 
class OpticksAttrSeq ;
template <typename> class OpticksCfg ;
class GGeo ; 
class GMesh ;
class GMergedMesh ;

class OpticksGeometry {
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

inline OpticksGeometry::OpticksGeometry(Opticks* opticks)
   :
   m_opticks(opticks),
   m_fcfg(NULL),
   m_ggeo(NULL),
   m_mesh0(NULL)
{
    init();
}

inline GGeo* OpticksGeometry::getGGeo()
{
   return m_ggeo ; 
}


