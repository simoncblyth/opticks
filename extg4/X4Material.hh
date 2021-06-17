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
#include <vector>
#include <string>
#include "plog/Severity.h"

class G4Material ; 
class G4MaterialPropertiesTable ; 
class GMaterial ; 

/**
X4Material
===========

**/

class X4_API X4Material
{
    public:
        static const plog::Severity LEVEL ; 
        static std::string Digest();
        static std::string Digest(const std::vector<G4Material*>& materials);
        static std::string Digest(const G4Material* material);
        static std::string Desc(const std::vector<G4Material*>& mtlist) ; 

        static size_t NumProp(G4MaterialPropertiesTable* mpt);
        static std::string DescProps(G4MaterialPropertiesTable* mpt, int wid);

    public:
        static GMaterial* Convert(const G4Material* material);
        static bool HasEfficiencyProperty(const G4MaterialPropertiesTable* mpt) ; 
       // static void       AddProperties(GMaterial* mat, const G4MaterialPropertiesTable* mpt);

    public:
        X4Material(const G4Material* material); 
        GMaterial* getMaterial();
    private:
        void init();
    private:
        const G4Material*                m_material ;  
        const G4MaterialPropertiesTable* m_mpt ; 
        bool                             m_has_efficiency ; 
        GMaterial*                       m_mat ; 
   
};

