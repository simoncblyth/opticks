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

#include "NBoundingBox.hpp"
#include <iostream>

int main(int, char** argv)
{
    NBoundingBox bb ; 

    glm::vec3 al(-10,-10,-10); 
    glm::vec3 ah(10,10,10) ; 

    glm::vec3 bl(-20,-20,-20); 
    glm::vec3 bh(0,5,10) ; 

    bb.update( al, ah );
    bb.update( bl, bh );

    std::cout << argv[0]
              << " : "
              << bb.description()
              << std::endl 
              ;


    return 0 ; 
}
