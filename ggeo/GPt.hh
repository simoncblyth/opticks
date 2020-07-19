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
#include "NGLM.hpp"
#include "GGEO_HEAD.hh"
#include "GGEO_API_EXPORT.hh"
 
/**
GPt
=====

Canonically instanciated in X4PhysicalVolume::convertNode
where instances are associated with the GVolume of the 
structural tree.

vectors of GPt instances are collected into GPts m_pts within GMergedMesh.
The GPts are persisted into the geocache which allows GParts creation 
to be deferred postcache. 

**/

struct GGEO_API GPt 
{
    int         lvIdx ; 
    int         ndIdx ; 
    int         csgIdx ; 
    std::string spec ; 
    glm::mat4   placement ; 

    GPt( int lvIdx_, int ndIdx_, int csgIdx_, const char* spec_, const glm::mat4& placement_ ); 
    GPt( int lvIdx_, int ndIdx_, int csgIdx_, const char* spec_ ); 

    void setPlacement( const glm::mat4& placement_ ); 

    std::string desc() const ; 

}; 

#include "GGEO_TAIL.hh"

