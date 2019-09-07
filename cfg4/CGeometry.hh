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

class OpticksHub ; 
class CG4 ; 
class Opticks ; 
template <typename T> class OpticksCfg ; 
class CDetector ; 
class CMaterialLib ; 
class CMaterialTable ; 
class CMaterialBridge ; 
class CSurfaceBridge ; 
class CSensitiveDetector ; 

#include "plog/Severity.h"
#include "CFG4_API_EXPORT.hh"

/**
CGeometry
===========

Canonical m_geometry instance is ctor resident of CG4.


1. init with CGDMLDetector, or CTestDetector when using "--test" option 
2. relies on creator(CG4) to call CGeometry::hookup(CG4* g4) giving the geometry to Geant4 


**/

class CFG4_API CGeometry 
{
       static const plog::Severity LEVEL ; 
   public:
       CGeometry(OpticksHub* hub, CSensitiveDetector* sd);
       bool hookup(CG4* g4);
       void postinitialize();   // invoked by CG4::postinitialize after Geant4 geometry constructed
   public:
       CMaterialLib*    getMaterialLib() const ;
       CDetector*       getDetector() const ;
       CMaterialBridge* getMaterialBridge() const ;
       CSurfaceBridge*  getSurfaceBridge() const ;
       const std::map<std::string, unsigned>& getMaterialMap() const ;        
   private:
       void init();
       void export_();
   private:
       OpticksHub*          m_hub ; 
       CSensitiveDetector*  m_sd ; 

       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       CDetector*           m_detector ; 
       CMaterialLib*        m_mlib ; 
       CMaterialTable*      m_material_table ; 
       CMaterialBridge*     m_material_bridge ; 
       CSurfaceBridge*      m_surface_bridge ; 

};



