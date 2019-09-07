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
class GSurfaceLib ;  

class CSurfaceTable ; 
class CSkinSurfaceTable ; 
class CBorderSurfaceTable ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4OpticalSurface ;

class CFG4_API CSurfaceBridge 
{
    public:
        CSurfaceBridge( GSurfaceLib* slib );

        unsigned getSurfaceIndex(const G4OpticalSurface* surf);    // G4OpticalSurface instance to 0-based Opticks surface index
        const char* getSurfaceName(unsigned int index);            // 0-based Opticks surface index to shortname
        const G4OpticalSurface* getG4Surface(unsigned int index);  // 0-based Opticks surface index to G4OpticalSurface

        void dump(const char* msg="CSurfaceBridge::dump");
        void dumpMap(const char* msg="CSurfaceBridge::dumpMap");
        bool operator()(const G4OpticalSurface* a, const G4OpticalSurface* b);
    private:
        void initMap(CSurfaceTable* stab);
    private:
        GSurfaceLib*         m_slib ; 
        CSkinSurfaceTable*   m_skin ; 
        CBorderSurfaceTable* m_border  ;
    private:
        std::map<const G4OpticalSurface*, unsigned> m_g4toix ; 
        std::map<unsigned int, std::string>         m_ixtoname ; 

};

#include "CFG4_TAIL.hh"

