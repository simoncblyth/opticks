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
#include "S_get_option.hh"

int main(int argc, char **argv)
{
    const char* size_default = "1200,800" ; 

    int stack = get_option<int>(argc, argv, "--stack", "3000" );   
    int width = get_option<int>(argc, argv, "--size,0", size_default ) ;   
    int height = get_option<int>(argc, argv, "--size,1", size_default ) ;   

    std::cout << " stack " << stack << std::endl ; 
    std::cout << " width [" << width << "]" << std::endl ; 
    std::cout << " height [" << height << "]" << std::endl ; 

    return 0 ; 
}


