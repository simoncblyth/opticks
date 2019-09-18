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

class Opticks ; 
class G4VPhysicalVolume ;
class G4LogicalVolume ;
class G4LogicalBorderSurface ;

#include "plog/Severity.h"

/**
CCheck
========

Recursively traverses a Geant4 geometry tree, 
checking integrity. Loosely follows access pattern
of G4GDMLWriter.

**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CCheck {
    public:
        static const plog::Severity LEVEL ; 
    public:
        CCheck(Opticks* ok, G4VPhysicalVolume* top);
    public:
        void checkSurf();
    private:
        void checkSurfTraverse(const G4LogicalVolume* const lv, const int depth);
    private:
        const G4LogicalBorderSurface* GetBorderSurface(const G4VPhysicalVolume* const pvol) ;
    private:
        Opticks*                       m_ok ; 
        G4VPhysicalVolume*             m_top ; 
};

#include "CFG4_TAIL.hh"


