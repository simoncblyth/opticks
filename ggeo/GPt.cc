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

#include <cstring>
#include <sstream>
#include <iomanip>

#include "GPt.hh"
#include "GLMFormat.hpp"


GPt::GPt( int lvIdx_, int ndIdx_, int csgIdx_, const char* spec_, const glm::mat4& placement_ )
    :
    lvIdx(lvIdx_),
    ndIdx(ndIdx_),
    csgIdx(csgIdx_),
    spec(strdup(spec_)),
    placement(placement_)
{
} 

GPt::GPt( int lvIdx_, int ndIdx_, int csgIdx_, const char* spec_ )
    :
    lvIdx(lvIdx_),
    ndIdx(ndIdx_),
    csgIdx(csgIdx_),
    spec(strdup(spec_)),
    placement(1.0f)
{
} 

void GPt::setPlacement( const glm::mat4& placement_ )
{
    placement = placement_ ;  
}


std::string GPt::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " lvIdx " << std::setw(4) << lvIdx
       << " ndIdx " << std::setw(7) << ndIdx
       << " csgIdx " << std::setw(7) << csgIdx
       << " spec " << std::setw(30) << spec
       << " placement " << gformat( placement )   
       ; 

    return ss.str(); 
}

 
