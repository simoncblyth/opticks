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
#include "SYSRAP_API_EXPORT.hh"

/** 
SOpticksKey
============

Used to communicate the geometry identity digest
to be used by Opticks instanciation (especually OpticksResource)
without changing the Opticks interface.

This is for example used with direct G4 to Opticks running, 
where the geometry is translated.

This class is needed because the OPTICKS_KEY in the environment 
is not always appropriate to use, eg when translating geometry which 
generates a new OPTICKS_KEY 

**/

class SYSRAP_API SOpticksKey 
{
     private:
       static const plog::Severity  LEVEL ; 
      public:
        static const char* G4LIVE ; 
        static const char* IDSTEM ; 
        static const char* IDFILE ; 
        static const char* IDSUBD ; 
        static int         LAYOUT ; 
        static const char* LAYOUT_ ; 
        static bool         IsSet();
        static SOpticksKey* GetKey();
        static bool         SetKey(const char* spec=nullptr) ;  
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
        const char* getIdPath(const char* base) const ; 
    public:
        std::string desc() const ; 
        bool isKeySource() const ;   // current executable is direct geocache creator
        std::string export_() const ;
    public:
        static bool IsLive() ;
        bool        isLive() const ; 
    private:
        SOpticksKey(const char* spec); 
        void setLive(bool live) ; 
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
        bool        m_live ; 
     


        static SOpticksKey* fKey ; 

};
