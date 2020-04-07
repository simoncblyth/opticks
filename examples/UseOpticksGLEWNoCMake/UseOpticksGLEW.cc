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
#include <GL/glew.h>

int main()
{

#ifdef GL_VERSION_1_1
    std::cout << "GL_VERSION_1_1" << std::endl ; 
#endif

#ifdef GL_VERSION_2_0
    std::cout << "GL_VERSION_2_0" << std::endl ; 
#endif

#ifdef GL_VERSION_3_0
    std::cout << "GL_VERSION_3_0" << std::endl ; 
#endif

#ifdef GL_VERSION_4_0
    std::cout << "GL_VERSION_4_0" << std::endl ; 
#endif

#ifdef GL_VERSION_4_5
    std::cout << "GL_VERSION_4_5" << std::endl ; 
#endif


    // all these are defined on mac, although only OpenGL upto 4.1(?) is supported

    return 0 ; 
}
