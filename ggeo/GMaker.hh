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

template <typename T> class NPY ;
class NTrianglesNPY ; 

#include <vector>
#include <cstddef>
#include <glm/fwd.hpp>

class Opticks ; 
#include "OpticksCSG.h"

class NCSG ; 
struct gbbox ; 
struct nnode ; 

class GBndLib ; 
class GMeshLib ; 
class GVolume ; 
class GMesh ; 

#include "plog/Severity.h"

/**

GMaker
=======

Only one canonical instance m_maker resides in GGeoTest 


**/

#include "GGEO_API_EXPORT.hh"
class GGEO_API GMaker {
       static const plog::Severity LEVEL ;  
       friend class GMakerTest ; 
    public:
        static std::string PVName(const char* shapename, int idx=-1);
        static std::string LVName(const char* shapename, int idx=-1);
   public:
       GMaker(Opticks* ok, GBndLib* blib, GMeshLib* meshlib );
   public:
       GVolume* make(unsigned int index, OpticksCSG_t typecode, glm::vec4& param, const char* spec);
   public:
       GMesh*   makeMeshFromCSG( NCSG* csg ) ; 
       GVolume* makeVolumeFromMesh( unsigned ndIdx, const GMesh* mesh ) const ; 
       GVolume* makeVolumeFromMesh( unsigned ndIdx, const GMesh* mesh, const glm::mat4& txf   ) const ; 
   private:
       void init();    

       static GVolume* makePrism(glm::vec4& param, const char* spec);
       static GVolume* makeBox(glm::vec4& param);
       static GVolume* makeZSphere(glm::vec4& param);
       static GVolume* makeZSphereIntersect_DEAD(glm::vec4& param, const char* spec);
       static void makeBooleanComposite(char shapecode, std::vector<GVolume*>& volumes,  glm::vec4& param, const char* spec);
       static GVolume* makeBox(gbbox& bbox);
   private:
       static GVolume* makeSubdivSphere(glm::vec4& param, unsigned int subdiv=3, const char* type="I");
       static NTrianglesNPY* makeSubdivSphere(unsigned int nsubdiv=3, const char* type="I");
       static GVolume* makeSphere(NTrianglesNPY* tris);
   private:
       Opticks*  m_ok ; 
       GBndLib*  m_bndlib ; 
       GMeshLib* m_meshlib ; 
};


