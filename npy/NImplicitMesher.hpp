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

#include <functional>
#include <vector>
#include <string>

#include "NGLM.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class BTimeKeeper ; 
class NTrianglesNPY ; 
class ImplicitMesherF ;
struct nnode ; 
struct nbbox ; 

class NPY_API NImplicitMesher
{
    public:
        typedef std::function<float(float,float,float)> FUNC ; 
    public:
        NImplicitMesher(nnode* node, int resolution=100, int verbosity=1, float expand_bb=1e-4, int ctrl=0, std::string seedstr="");
        NTrianglesNPY* operator()();
 
        void setFunc(FUNC sdf);

        NTrianglesNPY* sphere_test(); 
        std::string desc();
        void profile(const char* s);
        void report(const char* msg="NImplicitMesher::report");
    
    private:
        void init();
        int addSeeds();
        int addManualSeeds();
        int addCenterSeeds();
        NTrianglesNPY* collectTriangles(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::ivec3>& tris );

    private:
        BTimeKeeper*     m_timer ; 
        nnode*           m_node ; 
        nbbox*           m_bbox ; 
        FUNC             m_sdf ;  
        ImplicitMesherF* m_mesher ; 
        int              m_resolution; 
        int              m_verbosity ; 
        float            m_expand_bb ;  
        int              m_ctrl ;  
        std::string      m_seedstr ; 

};
