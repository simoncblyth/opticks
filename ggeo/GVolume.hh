/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <vector>
#include <string>

#include "NGLM.hpp"
class GMesh ;
class GParts ; 
struct GPt ; 

template <typename T> class GMatrix ; 

#include "OpticksCSG.h"
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
      GVolume( unsigned index, GMatrix<float>* transform, const GMesh* mesh, void* origin_node );
  public:
      void     setCSGFlag(OpticksCSG_t flag);
      void     setBoundary(unsigned boundary);     // also sets BoundaryIndices array
      void     setBoundaryAll(unsigned boundary);  // recursive over tree

  public:
      static const unsigned SENSOR_UNSET ; 
      void     setSensorIndex(unsigned sensorIndex) ;
      unsigned getSensorIndex() const ;
      bool     hasSensorIndex() const ;
  public:
      // CopyNumber comes from G4PVPlacement.CopyNo (physvol/@copynumber in GDML)
      void     setCopyNumber(unsigned copyNumber); 
      unsigned getCopyNumber() const ;  
  public:
      void        setPVName(const char* pvname);
      void        setLVName(const char* lvname);
      const char* getPVName() const ;
      const char* getLVName() const ;
  public:
      OpticksCSG_t getCSGFlag() const ;

      unsigned     getBoundary() const ;
      unsigned     getShapeIdentity() const ;
      glm::uvec4   getIdentity() const ;
      glm::uvec4   getNodeInfo() const ; 
  public:
      GParts*      getParts() const ;
      GPt*         getPt() const ;
      void         setParts(GParts* pts);
      void         setPt(GPt* pt);
  public:
      // ancillary slot for a parallel node tree, used by X4PhysicalVolume
      void*        getParallelNode() const ;
      void         setParallelNode(void* pnode); 
  public:
      // OriginNode records the G4VPhysicalVolume from whence the GVolume was converted, see X4PhysicalVolume::convertNode
      void*        getOriginNode() const ;
  public:
      void         setOuterVolume(const GVolume* outer_volume); 
      const GVolume*   getOuterVolume() const ; 
  public: 
      void Summary(const char* msg="GVolume::Summary");
      std::string description() const ;
  private:
      int               m_boundary ; 
      OpticksCSG_t      m_csgflag ; 
      unsigned          m_sensorIndex ; 

      const char*       m_pvname ; 
      const char*       m_lvname ; 

      GParts*           m_parts ; 
      GPt*              m_pt ; 
      void*             m_parallel_node ; 
      int               m_copyNumber ; 

      void*             m_origin_node ; 
      const GVolume*    m_outer_volume ; 


};
#include "GGEO_TAIL.hh"

