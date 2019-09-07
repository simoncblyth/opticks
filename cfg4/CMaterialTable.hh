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
#include <map>
#include "plog/Severity.h"

class G4Material ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CMaterialTable {
    public:
         static const plog::Severity LEVEL ;  
    public:
         CMaterialTable(const char* prefix="/dd/Materials/");
         void dump(const char* msg="CMaterialMap::dump");
         void fillMaterialIndexMap( std::map<std::string, unsigned>&  mixm );
         const std::map<std::string, unsigned>& getMaterialMap() const ;
    public:
         void dumpMaterial(unsigned index);
         unsigned getMaterialIndex(const char* shortname);
         void dumpMaterial(const char* shortname);
         void dumpMaterial(G4Material* material);
    private:
         void init();
         void initNameIndex();
    private:
         const char* m_prefix ; 
         std::map<std::string, unsigned> m_name2index ; 
         std::map<unsigned, std::string> m_index2name ; 

};

#include "CFG4_TAIL.hh"

