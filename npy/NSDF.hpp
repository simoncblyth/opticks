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
#include <functional>

#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"

/**
NSDF
=====

Applies inverse transform v to take global positions 
into frame of the structural nd 
where the CSG is placed, these positions local 
to the CSG can then be used with the CSG SDF to 
see the distance from the point to the surface of the 
solid.
 
TODO: investigate: currently only compiled when YoctoGL_FOUND but seems no dependency on that ?

**/

struct NPY_API NSDF
{
    typedef std::vector<float>::const_iterator VFI ; 

    NSDF(std::function<float(float,float,float)> sdf, const glm::mat4& inverse );
    float operator()( const glm::vec3& q_ );

    void clear(); 
    void classify( const std::vector<glm::vec3>& qq, float epsilon, unsigned expect, bool dump=false)  ;

    bool is_error() const ;
    bool is_empty() const ;
    std::string desc() const ;
    std::string detail() const ;

    std::function<float(float,float,float)> sdf ; 
    const glm::mat4                         inverse ; 
    unsigned                                verbosity ; 

    std::vector<float>                      sd ; 
    glm::uvec4                              tot ; 
    glm::vec2                               range ;                  

    // hang on to last classification prameters for dumping 
    float                                   epsilon ; 
    unsigned                                expect ; 
    const std::vector<glm::vec3>*           qqptr ; 
};

