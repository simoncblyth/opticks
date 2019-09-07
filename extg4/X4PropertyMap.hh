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

class G4MaterialPropertiesTable ; 
template <typename T> class GPropertyMap ; 
class GMaterialLib ; 

/**
X4PropertyMap
================

Convert Opticks GGeo property map into Geant4 material properties table


**/

class X4_API X4PropertyMap
{
    public:
        static G4MaterialPropertiesTable* Convert( const GPropertyMap<float>* pmap );
    public:
        X4PropertyMap(const GPropertyMap<float>* pmap) ; 
        G4MaterialPropertiesTable* getMPT() const ;
    private:
        void init(); 
    private:
        const GPropertyMap<float>*   m_pmap ; 
        G4MaterialPropertiesTable*   m_mpt ;   
        const GMaterialLib*          m_mlib ; 
};


