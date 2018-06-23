#pragma once

//  hmm the difference between the models (Chroma and G4 at the time) is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//
// Instances are created by:
//       GVolume* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GVolume* parent)
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

GVolume
========

GVolume was formerly mis-named as "GSolid", which was always a confusing name.

Constituent "solids" (as in CSG, and G4VSolid in G4) are mesh-level-objects 
(relatively few in a geometry, corresponding to each distinct shape)

Whereas GVolume(GNode) are node-level-objects (relatively many in the geometry)
which refer to the corresponding mesh-level-objects by pointer or index.

**/

class GGEO_API GVolume : public GNode {
  public:
      static void Dump( const std::vector<GVolume*>& solids, const char* msg="GVolume::Dump" );
  public:
      GVolume( unsigned int index, GMatrix<float>* transform, const GMesh* mesh,  unsigned int boundary, NSensor* sensor);
  public:
      void setCSGFlag(OpticksCSG_t flag);
      void setCSGSkip(bool csgskip);
      void setBoundary(unsigned boundary);     // also sets BoundaryIndices array
      void setBoundaryAll(unsigned boundary);  // recursive over tree
      void setSensor(NSensor* sensor);         // also sets SensorIndices
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
      // ancillary slot for a parallel node tree, used by X4PhysicalVolume
      void*        getParallelNode() const ;
      void         setParallelNode(void* pnode); 
  public: 
      void Summary(const char* msg="GVolume::Summary");
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
      void*             m_parallel_node ; 

};
#include "GGEO_TAIL.hh"

