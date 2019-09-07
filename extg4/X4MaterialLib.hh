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

template <typename T> class GPropertyMap ; 
class GMaterialLib ;
#include "G4MaterialTable.hh"

/**
X4MaterialLib
================

NB not a full conversion, just replaces Geant4 material MPT with 
the standardized domain properties from the Opticks GMaterialLib

**/

class X4_API X4MaterialLib
{
    public:
        static void Standardize(); 
        static void Standardize( G4MaterialTable* mtab, const GMaterialLib* mlib ); 
    public:
        X4MaterialLib(G4MaterialTable* mtab,  const GMaterialLib* mlib) ; 
    private:
        void init(); 
    private:
        G4MaterialTable*      m_mtab ; 
        const GMaterialLib*   m_mlib ; 
};


