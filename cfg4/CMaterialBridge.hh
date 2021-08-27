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
#include "plog/Severity.h"
class GMaterialLib ;  

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4StepPoint ; 
class G4Step ; 
class G4Material ; 

/**
CMaterialBridge
=================

Provides a mapping between G4 and Opticks materials, 
the Opticks material lib needs to be closed before 
instanciation of CMaterialBridge.

The mapping is based in matching matial shortnames 
between the two models.

**/

class CFG4_API CMaterialBridge 
{
        static const plog::Severity LEVEL ; 
    public:
        CMaterialBridge(const GMaterialLib* mlib );

        unsigned getPointMaterial(const G4StepPoint* point) const ;
        unsigned getPreMaterial(const G4Step* step) const ;
        unsigned getPostMaterial(const G4Step* step) const ;

        unsigned getMaterialIndex(const G4Material* mat) const ; // G4Material instance to 0-based Opticks material index
        const char* getMaterialName(unsigned int index, bool abbrev=true) const ;  // 0-based Opticks material index to shortname
        const G4Material* getG4Material(unsigned int index) const ; // 0-based Opticks material index to G4Material

        std::string MaterialSequence(unsigned long long seqmat, bool abbrev=true ) const ;

        void dump(const char* msg="CMaterialBridge::dump") const ;
        void dumpMap(const char* msg="CMaterialBridge::dumpMap") const ;
        bool operator()(const G4Material* a, const G4Material* b);

        bool isValid() const ; 
    private:
        void initMap();
    private:
        const GMaterialLib*   m_mlib ; 
        const Opticks* m_ok ; 
        const bool m_test ; 
        const unsigned m_mlib_materials ; 
        const unsigned m_g4_materials ; 

        std::map<const G4Material*, unsigned> m_g4toix ; 
        std::map<unsigned int, std::string>   m_ixtoname ; 
        std::map<unsigned int, std::string>   m_ixtoabbr ; 

};

#include "CFG4_TAIL.hh"

