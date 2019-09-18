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

// op --cgdmldetector
#pragma once


class OpticksQuery ; // okc-
class OpticksHub ;   // okg-
class G4VPhysicalVolume ; 


class CSensitiveDetector ; 

#include "CDetector.hh"
#include "CFG4_API_EXPORT.hh"
#include "plog/Severity.h"

/**
CGDMLDetector
==============

*CGDMLDetector* is a :doc:`CDetector` subclass that
loads Geant4 GDML persisted geometry files, 
from m_ok->getGDMLPath().

**/

class CFG4_API CGDMLDetector : public CDetector
{
  public:
    static const plog::Severity LEVEL ;  
  public:
    CGDMLDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd);
    void saveBuffers();
    virtual ~CGDMLDetector();
  private:
    void init();
  private:
    G4VPhysicalVolume* parseGDML(const char* path) const ;

    void sortMaterials();
    void addMPTLegacyGDML();
    void standardizeGeant4MaterialProperties(); // by adoption of those from Opticks  


    //void addSD();    <-- too early SD only gets created later at CG4 
    void addSurfaces();
    //void kludge_cathode_efficiency();


};


