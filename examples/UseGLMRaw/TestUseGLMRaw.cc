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


#include <iostream>
#include "UseGLMRaw.hh"

#include <glm/gtx/string_cast.hpp>

int main(int argc, char** argv)
{

    float tr = 100.f ; 
    glm::vec2 rot(10,10) ; 

    glm::mat4 pvm = camera( tr, rot ); 


    std::cout << glm::to_string(pvm) << std::endl ; 


    return 0 ; 
}
