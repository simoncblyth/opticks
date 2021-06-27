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

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 

/**
X4MaterialPropertiesTable 
===========================

Converts properties from G4MaterialPropertiesTable into the
GPropertyMap<float> base of GMaterial, GSkinSurface or GBorderSurface.

The _OLD methods give runtime warnings with 10.4.2 threatening that 
map accessors will be removed in 11

**/

class X4_API X4MaterialPropertiesTable 
{
        static const plog::Severity LEVEL ; 
    public:
        static void Convert(GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt);
        static std::string Digest(const G4MaterialPropertiesTable* mpt);
        static std::string Digest_OLD(const G4MaterialPropertiesTable* mpt);
    private:
        X4MaterialPropertiesTable(GPropertyMap<double>* pmap,  const G4MaterialPropertiesTable* const mpt);
        void init();
    private:
        static void AddProperties(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt);
        static void AddProperties_OLD(GPropertyMap<double>* pmap, const G4MaterialPropertiesTable* const mpt);
    private:
        GPropertyMap<double>*                   m_pmap ; 
        const G4MaterialPropertiesTable* const m_mpt ;

};
