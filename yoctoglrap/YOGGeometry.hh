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

/**
YOG::Geometry
==============

Holder of vertices and indices.


**/


#include "YOG_API_EXPORT.hh"
#include <vector>

template <typename T> class NPY ;

namespace YOG  {
struct YOG_API Geometry
{
    Geometry( int count_ );

    int             count ; 
    NPY<float>*     vtx ; 
    NPY<unsigned>*  idx ; 

    std::vector<float> vtx_minf ; 
    std::vector<float> vtx_maxf ; 

    std::vector<unsigned> idx_min ; 
    std::vector<unsigned> idx_max ; 

    std::vector<float> idx_minf ; 
    std::vector<float> idx_maxf ; 


    void make_triangle(); 
};

}  // namespace

