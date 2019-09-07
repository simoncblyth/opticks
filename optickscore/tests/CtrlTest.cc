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

// TEST=CtrlTest om-t
#include <cassert>

#include "NGLM.hpp"
#include "NPY.hpp"
#include "Ctrl.hh"

#include "OPTICKS_LOG.hh"


union u_f4_c16
{
     float f[4] ;
     char  c[16] ;
};

void test_vec4()
{
    u_f4_c16 fc ;
    memset( fc.c, 0, 16 );

    fc.c[0] = 'c' ; 
    fc.c[1] = '0' ; 

    fc.c[2] = 'c' ; 
    fc.c[3] = '1' ; 

    fc.c[4] = 'c' ; 
    fc.c[5] = '2' ; 


    glm::vec4 v ;
    v.x = fc.f[0] ;
    v.y = fc.f[1] ;
    v.z = fc.f[2] ;
    v.w = fc.f[3] ;

    Ctrl ctrl(glm::value_ptr(v), 4);
    std::cout << ctrl.getCommands() << std::endl ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_vec4() ; 


    return 0 ;
}

