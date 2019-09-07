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

class GGeo ; 
class GMaterialLib ; 
class Opticks ; 
class G4Material ; 

#include "G4MaterialTable.hh"   // typedef std::vector<G4Material*>
#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

/*
X4MaterialTable
===========================

void X4MaterialTable::Convert(GMaterialLib* mlib) 
---------------------------------------------------

Converts all G4Material from the static G4MaterialTable 
into Opticks GMaterial and adds them to the GMaterialLib
in original creation order with original indices.

* mlib is expected to be empty when this is called.

*/

class X4_API X4MaterialTable 
{
    public:
        static const plog::Severity  LEVEL ; 
        static void Convert(GMaterialLib* mlib) ; 
        static G4Material* Get(unsigned idx);
    private:
        X4MaterialTable(GMaterialLib* mlib);
        GMaterialLib* getMaterialLib();
        void init();
    private:
        const G4MaterialTable*  m_mtab ; 
        GMaterialLib*           m_mlib ;         
};

