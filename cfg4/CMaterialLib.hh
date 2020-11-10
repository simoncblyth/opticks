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

#include <map>
#include <string>
#include "plog/Severity.h"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

template <typename T> class NPY ; 

class OpticksHub ; 

class GMaterial ; 
class G4Material ; 

class CMPT ; 
#include "CPropLib.hh"

/**
CMaterialLib
===============

CMaterialLib is a constituent of CDetector (eg CTestDector and CGDMLDetector)
that is instanciated with it.
that converts GGeo (ie Opticks G4DAE) materials and surfaces into G4 materials and surfaces.
G4The GGeo gets loaded on initializing base class CPropLib.

SUSPECT THIS CAN BE REMOVED, IN DIRECT APPROACH THERE SHOULD NOT
BE MUCH NEED FOR IT 


WAS SURPRISED TO FIND THAT THE CONVERSION IS NOT DONE BY STANDARD LAUNCH
INSTEAD THE INDIVIDUAL convertMaterial ARE CALLED FROM EG CTestDetector 
WHICH POPULATES THE MAP.

**/


class CFG4_API CMaterialLib : public CPropLib 
{
   public:
       static const plog::Severity LEVEL ; 
   public:
       CMaterialLib(OpticksHub* hub);

       bool isConverted();

       void convert(); // commented in init, never invoked in standard running 
       void postinitialize();  // invoked from CGeometry::postinitialize 

       const G4Material* makeG4Material(const char* matname);
       const G4Material* convertMaterial(const GMaterial* kmat);

       void dump(const char* msg="CMaterialLib::dump");
       void saveGROUPVEL(const char* base="$TMP/CMaterialLib");

       // G4 material access
       bool hasG4Material(const char* shortname);
       const G4Material* getG4Material(const char* shortname) const ;
       const CMPT*       getG4MPT(const char* shortname) const ;


       void dumpGroupvelMaterial(const char* msg, float wavelength, float groupvel, float tdiff, int step_id, const char* qwn="" );
       std::string firstMaterialWithGroupvelAt430nm(float groupvel, float delta=0.0001f);
       void fillMaterialValueMap(const char* matnames);
       void fillMaterialValueMap(std::map<std::string,float>& vmp,  const char* matnames, const char* key, float nm);
       static void dumpMaterialValueMap(const char* msg, std::map<std::string,float>& vmp);
       static std::string firstKeyForValue(float val, std::map<std::string,float>& vmp, float delta=0.0001f );
       

       const G4Material* getG4Material(unsigned index);
       NPY<float>* makeArray(const char* name, const char* keys, bool reverse=true);
   private:
       void dump(const GMaterial* mat, const char* msg="CMaterialLib::dump");
       void dumpMaterials(const char* msg="CMaterialLib::dumpMaterials");
       void dumpMaterial(const G4Material* mat, const char* msg="CMaterialLib::dumpMaterial");

   private:
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);

   private:
       bool                                            m_converted ;      
       std::map<const GMaterial*, const G4Material*>   m_ggtog4 ; 
       std::map<std::string, const G4Material*>        m_g4mat ; 
   private:
       std::map<std::string, float>                    m_groupvel_430nm ; 


};

#include "CFG4_TAIL.hh"

