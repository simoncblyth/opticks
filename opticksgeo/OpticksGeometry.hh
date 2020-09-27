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

#include <string>
#include <map>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

class OpticksHub ; 
class Opticks ; 
class Composition ; 
template <typename> class OpticksCfg ;
class GGeo ; 
class GMesh ;
class GMergedMesh ;

/**
OpticksGeometry : GGeo holder/loader/fixer 
============================================

* almost nothing here is needed anymore (meshfixing and Assimp loading no longer viable)
* TODO: eliminate this just go direct to GGeo ?


Actually OpticksGGeo would be a better name, this acts as a higher 
level holder of GGeo with a triangulated (G4DAE) focus.  
Anything related to analytic (GLTF) should not live here, OpticksHub 
would be more appropriate.

* ACTUALLY ARE NOW THINKING THAT ANALYTIC SHOULD LIVE INSIDE
  A SINGLE GGeo ALONGSIDE THE TRIANGULATED 

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
       static const plog::Severity LEVEL ; 
   public:
       OpticksGeometry(OpticksHub* hub);
  public:
       void loadGeometry();
  public:
       GGeo*           getGGeo();
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


