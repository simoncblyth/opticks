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


#include "GLMPrint.hpp"

#include "NTriangle.hpp"
#include "PLOG.hh"

ntriangle::ntriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) 
{
    p[0] = a ; 
    p[1] = b ; 
    p[2] = c ; 
}

ntriangle::ntriangle(float* ptr)
{
    p[0] = glm::make_vec3(ptr) ; 
    p[1] = glm::make_vec3(ptr+3) ; 
    p[2] = glm::make_vec3(ptr+6) ; 
}    

void ntriangle::copyTo(float* ptr) const 
{
    memcpy( ptr+0, glm::value_ptr(p[0]), sizeof(float)*3 );
    memcpy( ptr+3, glm::value_ptr(p[1]), sizeof(float)*3 );
    memcpy( ptr+6, glm::value_ptr(p[2]), sizeof(float)*3 );
}

void ntriangle::dump(const char* msg)
{
    LOG(info) << msg ; 
    print(p[0], "p[0]");
    print(p[1], "p[1]");
    print(p[2], "p[2]");
}



