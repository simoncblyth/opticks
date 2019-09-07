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
#include <glm/fwd.hpp>

// okc-
class OpticksHub ; 
class OpticksQuery ; 

// npy-
struct nnode ; 
class NCSG ; 
class NGeoTestConfig ; 

// ggeo-
class GGeoTest ; 
class GMaterial ;
class GCSG ; 

// cfg4-
class CPropLib ; 
class CSensitiveDetector ; 

// g4-
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VSolid;


#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"
#include "plog/Severity.h"

/**

CTestDetector
=================

*CTestDetector* is a :doc:`CDetector` subclass that
constructs simple Geant4 detector test geometries based on commandline specifications
parsed and represented by an instance of :doc:`../npy/NGeoTestConfig`.

Canonical instance resides in CGeometry and is instanciated by CGeometry::init
when --test option is used. After the instanciation the CDetector::attachSurfaces
is invoked.

**/


class CFG4_API CTestDetector : public CDetector
{
    static const plog::Severity LEVEL ; 
 public:
    CTestDetector(OpticksHub* hub, OpticksQuery* query=NULL, CSensitiveDetector* sd=NULL);
  private:
    void init();

  private:
    G4VPhysicalVolume* makeDetector();
    G4VPhysicalVolume* makeDetector_NCSG();
    void boxCenteringFix( glm::vec3& placement, nnode* root  );
    G4VPhysicalVolume* makeChildVolume(const NCSG* csg, const char* lvn, const char* pvn, G4LogicalVolume* mother, const NCSG* altcsg );
    G4VPhysicalVolume* makeVolumeUniverse(const NCSG* csg);

  private:
    GGeoTest*          m_geotest ; 
    NGeoTestConfig*    m_config ; 

};



