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

// TODO: migrate to optickscore-
// attack class to kill Types


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/* 
Typ
====

Setup from GGeo::setupTyp GGeo::loadGeometry

*/


class NPY_API Typ {
    public:
        Typ();
        void setMaterialNames(std::map<unsigned int, std::string> material_names);
        void setFlagNames(std::map<unsigned int, std::string> flag_names);
        std::string findMaterialName(unsigned int);

        void dump(const char* msg="Typ::dump") const ;
        void dumpMap(const char* msg, const std::map<unsigned, std::string>& m ) const ;

    private:
        std::map<unsigned int, std::string> m_material_names ; 
        std::map<unsigned int, std::string> m_flag_names ; 

};

#include "NPY_TAIL.hh"

