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

#include "plog/Severity.h"

class OpticksHub ; 
class Opticks ; 
class Composition ; 
class GGeo ; 

/**
OpticksGeometry : GGeo holder/loader/fixer 
============================================

* almost nothing here is needed anymore (meshfixing and Assimp loading no longer viable)
* TODO: eliminate this just go direct to GGeo 

Canonical m_geometry instance resides in okg/OpticksHub 
and is instanciated by OpticksHub::init which 
happens within the ok/OKMgr or okg4/OKG4Mgr ctors.

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
       void init();
   private:
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       Composition*         m_composition ; 
       GGeo*                m_ggeo ; 
       unsigned             m_verbosity ;
};


