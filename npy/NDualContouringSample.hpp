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
#include "NBBox.hpp"
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"

class BTimeKeeper ; 
class NTrianglesNPY ; 
struct nnode ; 

class NPY_API NDualContouringSample 
{
    public:
        NDualContouringSample(int nominal=7, int coarse=6, int verbosity=1, float threshold=0.1f, float scale_bb=1.01f );
        NTrianglesNPY* operator()(nnode* node); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NDualContouringSample::report");
    private:
        BTimeKeeper* m_timer ; 
        int    m_nominal; 
        int    m_coarse; 
        int    m_verbosity ; 
        float  m_threshold ; 
        float  m_scale_bb ; 

};
