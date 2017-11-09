#pragma once

//  hmm the difference between the models (Chroma and G4 at the time) is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//
// Instances are created by:
//       GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
//


#include <vector>
#include <string>

class GMesh ;
class GParts ; 
class NSensor ; 
template <typename T> class GMatrix ; 

#include "OpticksCSG.h"
#include "GVector.hh"

#include "GNode.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**

GSolid
========

TODO: restructure

* perhaps just rename GSolid -> GVolume(?) or GVolNode 
* OR move everythng down into GNode ?
* OR rename current GNode -> GVNode making it pure virtual and use the GNode name up here ?

* GSolid is a node-level-object, not a mesh-level-object but the name GSolid implies a mesh-level-object
* GSolid just adds a few properties to its GNode base class


**/

class GGEO_API GSolid : public GNode {
  public:
      static void Dump( const std::vector<GSolid*>& solids, const char* msg="GSolid::Dump" );
  public:
      GSolid( unsigned int index, GMatrix<float>* transform, const GMesh* mesh,  unsigned int boundary, NSensor* sensor);
  public:
  public:
      void setCSGFlag(OpticksCSG_t flag);
      void setCSGSkip(bool csgskip);
      void setBoundary(unsigned boundary);
      void setBoundaryAll(unsigned boundary);
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
      OpticksCSG_t getCSGFlag();
      bool         isCSGSkip();
      unsigned int getBoundary() const ;
      guint4       getIdentity();
      //void  setIdentity(const guint4& id );

      NSensor*     getSensor();
  public:
      GParts*      getParts();
      void         setParts(GParts* pts);
  public: 
      void Summary(const char* msg="GSolid::Summary");
      std::string description();
  private:
      unsigned int      m_boundary ; 
      OpticksCSG_t      m_csgflag ; 
      bool              m_csgskip ; 
      NSensor*          m_sensor ; 
      const char*       m_pvname ; 
      const char*       m_lvname ; 
      unsigned int      m_sensor_surface_index ; 
      GParts*           m_parts ; 

};
#include "GGEO_TAIL.hh"

