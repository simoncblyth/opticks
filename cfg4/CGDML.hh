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
#include "plog/Severity.h"
#include "CFG4_API_EXPORT.hh"

/**
**/

class G4GDMLParser ; 
class G4VPhysicalVolume ; 
class NMeta ; 


class CFG4_API CGDML
{
    private:
        static const plog::Severity LEVEL ; 
        static G4GDMLParser* InitParser(const char* path);
    public:
        static G4VPhysicalVolume* Parse(const char* path);
        static G4VPhysicalVolume* Parse(const char* path, NMeta** meta);
    public:
        static void Export(const char* dir, const char* name, const G4VPhysicalVolume* const world );
        static void Export(const char* path, const G4VPhysicalVolume* const world );
        static std::string GenerateName(const char* name, const void* const ptr, bool addPointerToName=true );
    public:
        CGDML(const char* path); 
    public:
        G4VPhysicalVolume*  getWorldVolume() const ; 
        NMeta*              getAuxMeta() const ; 
        void                dumpAux(const char* msg="CGDML::dumpAux") const ; 
    private:
        G4GDMLParser*       m_parser  ;


};


 
