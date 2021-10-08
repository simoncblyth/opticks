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
#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include "G4String.hh"

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 

/**
X4MaterialPropertiesTable 
===========================

Converts properties from G4MaterialPropertiesTable into the
GPropertyMap<double> base of GMaterial, GSkinSurface or GBorderSurface.

**/

class X4_API X4MaterialPropertiesTable 
{
        static const plog::Severity LEVEL ; 
    public:
        static G4MaterialPropertyVector* 
                    GetProperty(           const G4MaterialPropertiesTable* mpt, const char* key ); 

        static int  GetPropertyIndex(      const G4MaterialPropertiesTable* mpt, const char* key ); 
        static int  GetConstPropertyIndex( const G4MaterialPropertiesTable* mpt, const char* key ); 
        static int  GetIndex(              const std::vector<G4String>& names,   const char* key ); 
        static bool PropertyExists(        const G4MaterialPropertiesTable* mpt, const char* key );
        static bool ConstPropertyExists(   const G4MaterialPropertiesTable* mpt, const char* key );

        static void Dump( const G4MaterialPropertiesTable* mpt, bool all ); 
        static void DumpPropMap( const G4MaterialPropertiesTable* mpt, bool all ); 
        static void DumpConstPropMap( const G4MaterialPropertiesTable* mpt, bool all ); 

    public:
        static void Convert(GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, char mode );
        static std::string Digest(const G4MaterialPropertiesTable* mpt);
    private:
        X4MaterialPropertiesTable(GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt, char mode );
        void init();
    private:
        static void AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt, char mode );
    private:
        GPropertyMap<double>*                  m_pmap ; 
        const G4MaterialPropertiesTable* const m_mpt ;
        char                                   m_mode ;   // 'G':G4 interpolate (should be best?)   'S':old-pmap-standardized    or 'A':asis  

};
