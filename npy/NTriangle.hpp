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

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

struct NPY_API ntriangle 
{
    ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
    ntriangle(float* ptr);
    void copyTo(float* ptr) const;
    void dump(const char* msg="ntriangle::dump");

    glm::vec3 p[3];
}; 

#include "NPY_TAIL.hh"


