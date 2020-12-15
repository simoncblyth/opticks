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
class BMeta ; 


class CFG4_API CGDML
{
    private:
        friend struct CGDMLTest ; 
    private:
        static const char* LVMETA ; 
        static const char* USERMETA ; 
    private:
        static const plog::Severity LEVEL ; 
    public:
        static G4VPhysicalVolume* Parse(const char* path);
        static G4VPhysicalVolume* Parse(const char* path, BMeta** meta );
    public:
        static G4VPhysicalVolume* Parse(const char* dir, const char* name);
        static G4VPhysicalVolume* Parse(const char* dir, const char* name, BMeta** meta );
    public:
        static void Export(const char* dir, const char* name, const G4VPhysicalVolume* const world, const BMeta* meta=NULL );
        static void Export(const char* path,                  const G4VPhysicalVolume* const world, const BMeta* meta=NULL );
        static std::string GenerateName(const char* name, const void* const ptr, bool addPointerToName=true );
    public:
        CGDML(); 
        void read(const char* path);  
        void write(const char* path,  const G4VPhysicalVolume* const world, const BMeta* meta=NULL );
    public:
        G4VPhysicalVolume*  getWorldVolume() const ; 
        BMeta*              getMeta() const ; 
        void                addMeta(const BMeta* meta);
    private:
        void                addLVMeta(const BMeta* lvmeta);
        void                addUserMeta(const BMeta* user);
    private:
        BMeta*              getLVMeta() const ; 
        BMeta*              getUserMeta() const ; 
        void                dumpLVMeta(const char* msg="CGDML::dumpLVMeta") const ; 
        void                dumpUserMeta(const char* msg="CGDML::dumpLVMeta") const ; 
    private:
        G4GDMLParser*       m_parser  ;
        bool                m_write_refs ; 
        const char*         m_write_schema_location ; 
        bool                m_read_validate ; 
        bool                m_read_trimPtr ; 
};


 
