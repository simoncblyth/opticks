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

#include "InterpolatedView.hh"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    View* a = new View ;   a->setEye(1,0,0) ;
    View* b = new View ;   b->setEye(-1,0,0) ;
  
    InterpolatedView* iv = new InterpolatedView ; 
    iv->addView(a);
    iv->addView(b);

    glm::mat4 m2w ; 
    print(m2w, "m2w");    

    unsigned int n = 10 ; 
    for(unsigned int i=0 ; i < n ; i++)
    {
        float f = float(i)/float(n) ;
        iv->setFraction(f);
        std::cout << "f " << f ; 
        glm::vec4 e = iv->getEye(m2w);
        print(e, "eye");    
    }


    return 0 ; 
}
