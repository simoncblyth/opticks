#pragma once

//  hmm the difference between the models (Chroma and G4 at the time) is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//
// Instances are created by:
//       GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
//


#include <string>

class GMesh ;
class GParts ; 
class NSensor ; 
template <typename T> class GMatrix ; 

#include "GVector.hh"

#include "GNode.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"
class GGEO_API GSolid : public GNode {
  public:
      GSolid( unsigned int index, GMatrix<float>* transform, GMesh* mesh,  unsigned int boundary, NSensor* sensor);
  public:
      void setSelected(bool selected);
      bool isSelected();
  public:
      void setBoundary(unsigned int boundary);
      void setSensor(NSensor* sensor);
      void setSensorSurfaceIndex(unsigned int ssi);
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
      GParts*      getParts();
      void         setParts(GParts* pts);
  public: 
      void Summary(const char* msg="GSolid::Summary");
      std::string description();
  private:
      unsigned int      m_boundary ; 
      NSensor*          m_sensor ; 
      bool              m_selected ;
      const char*       m_pvname ; 
      const char*       m_lvname ; 
      unsigned int      m_sensor_surface_index ; 

};
#include "GGEO_TAIL.hh"

