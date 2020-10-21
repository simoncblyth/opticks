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
#include "BRAP_API_EXPORT.hh"

/** 
BOpticksKey
============

Used to communicate the geometry identity digest
to be used by Opticks instanciation (especually OpticksResource)
without changing the Opticks interface.

This is for example used with direct G4 to Opticks running, 
where the geometry  


**/

class BRAP_API BOpticksKey 
{
     private:
       static const plog::Severity  LEVEL ; 
      public:
        static const char* G4LIVE ; 
        static const char* IDSTEM ; 
        static const char* IDFILE ; 
        static const char* IDSUBD ; 
        static int         LAYOUT ; 
        static bool         IsSet();
        static BOpticksKey* GetKey();
        static bool         SetKey(const char* spec) ;  
        static void         Desc() ;  
        static const char* StemName( const char* ext, const char* sep="." );
    public:
        const char* getSpec() const ; 
        const char* getExename() const ; 
        const char* getClass() const ; 
        const char* getVolname() const ; 
        const char* getDigest() const ; 
    public:
        const char* getIdname() const ; 
        const char* getIdfile() const ; 
        const char* getIdGDML() const ; 
        const char* getIdsubd() const ; 
        int         getLayout() const ; 
    public:
        std::string desc() const ; 
        bool isKeySource() const ;   // current executable is direct geocache creator
    private:
        BOpticksKey(const char* spec); 
    private:
        const char* m_spec   ; 

        const char* m_exename ; 
        const char* m_class   ; 
        const char* m_volname   ; 
        const char* m_digest ;

        const char* m_idname ; // eg OpNovice_World_g4live
        const char* m_idfile ; // eg g4ok.gltf
        const char* m_idgdml ; // eg g4ok.gdml
        const char* m_idsubd ; // eg g4ok_gltf
        int         m_layout ; 
        const char* m_current_exename ;
 
        static BOpticksKey* fKey ; 

};
