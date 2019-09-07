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

#include <vector>
#include <string>
#include <map>

class G4Material ; 
#include "G4MaterialTable.hh"
#include "CFG4_API_EXPORT.hh"

/**
CMaterialSort
==============

Sort the G4MaterialTable into the order specified
by the ctor argument.

BUT that makes the position of the G4Material
inconsistent with the fIndexInTable set in ctor and 
used from dtor::

    253   // Remove this material from theMaterialTable.
    254   //
    255   theMaterialTable[fIndexInTable] = 0;
    256 }

This means will be nullification of the wrong object 
on deleting G4Material.  But as there is not much 
reason to ever delete a material... lets see if it causes
any issue.

**/

class CFG4_API CMaterialSort {
        typedef std::map<std::string, unsigned> MSU ; 
   public:
        CMaterialSort(const std::map<std::string, unsigned>& order ); 
        bool operator()(const G4Material* a, const G4Material* b) ;
   private:
        void init();      
        void dump(const char* msg) const ;      
        void dumpOrder(const char* msg) const ;      
        void sort();      
   private:
        const std::map<std::string, unsigned>&  m_order  ;
        G4MaterialTable*                        m_mtab ; 
        bool                                    m_dbg ; 
};

 
