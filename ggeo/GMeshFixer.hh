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
class GMesh ; 

/*
GMeshFixer
=============

In production it makes more sense to fix meshes earlier 
ie before the cache and even before GGeo gets involved at 
assimp import level. 

BUT during development it is convenient to operate
at a later stage as it is then easy to visualize the meshes.

*/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GMeshFixer {
    public:
        GMeshFixer(GMesh* src);
        ~GMeshFixer();
        void copyWithoutVertexDuplicates();

        GMesh* getSrc();
        GMesh* getDst();

    private:
        void mapVertices();
        void copyDedupedVertices();

    private:
        GMesh* m_src ; 
        GMesh* m_dst ; 

        int*   m_old2new ; 
        int*   m_new2old ; 

        unsigned int m_num_deduped_vertices ; 

        std::map<std::string, unsigned int> m_vtxmap ;

};

#include "GGEO_TAIL.hh"


