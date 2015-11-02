#pragma once

class GGeo ; 
class GBndLib ; 
class GSurfaceLib ; 
class GMesh ;
class NSensor ; 

#include "GNode.hh"
#include "GMatrix.hh"
#include "GVector.hh"
#include <climits>

//  hmm the difference between the models is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//
//
// Instances are created by:
//       GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
//
class GSolid : public GNode {
  public:
      GSolid( GGeo* ggeo, unsigned int index, GMatrixF* transform, GMesh* mesh,  unsigned int boundary, NSensor* sensor);
  private:  
      void init(); 
  public:
      void setSelected(bool selected);
      bool isSelected();
  public:
      void setBoundary(unsigned int boundary);
      void setSensor(NSensor* sensor);
      unsigned int getSensorSurfaceIndex();
  public:
      // need to resort to names for debugging IAV top lid issue
      void setPVName(const char* pvname);
      void setLVName(const char* lvname);
      const char* getPVName();
      const char* getLVName();
  public:
      unsigned int getBoundary();
      guint4       getIdentity();
      NSensor*     getSensor();
  public: 
      void Summary(const char* msg="GSolid::Summary");
  private:
      unsigned int      m_boundary ; 
      NSensor*          m_sensor ; 
      bool              m_selected ;
      const char*       m_pvname ; 
      const char*       m_lvname ; 
      GBndLib*          m_blib ; 
      GSurfaceLib*      m_slib ; 

};

inline GSolid::GSolid( GGeo* ggeo, unsigned int index, GMatrixF* transform, GMesh* mesh, unsigned int boundary, NSensor* sensor)
         : 
         GNode(ggeo, index, transform, mesh ),
         m_boundary(boundary),
         m_sensor(sensor),
         m_selected(true),
         m_pvname(NULL),
         m_lvname(NULL),
         m_blib(NULL),
         m_slib(NULL)
{
    init();
}


inline unsigned int GSolid::getBoundary()
{
    return m_boundary ; 
}


inline NSensor* GSolid::getSensor()
{
    return m_sensor ; 
}

inline void GSolid::setSelected(bool selected)
{
    m_selected = selected ; 
}
inline bool GSolid::isSelected()
{
   return m_selected ; 
}

inline void GSolid::setPVName(const char* pvname)
{
    m_pvname = strdup(pvname);
}
inline void GSolid::setLVName(const char* lvname)
{
    m_lvname = strdup(lvname);
}

inline const char* GSolid::getPVName()
{
    return m_pvname ; 
}
inline const char* GSolid::getLVName()
{
    return m_lvname ; 
}




