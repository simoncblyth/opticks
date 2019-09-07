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

#include <cstdlib>
#include "NGLMExt.hpp"

#include "NTorus.hpp"
#include "OPTICKS_LOG.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    float r = 10.f ; 
    float R = 100.f ; 

    ntorus* _torus = make_torus( 0, 0, 10, 100) ; 
    const ntorus& torus = *_torus ; 
 
    float epsilon = 1e-5 ; 

    assert( fabsf(torus(R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(R-r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R+r, 0, 0)) < epsilon );
    assert( fabsf(torus(-R-r, 0, 0)) < epsilon );



    


    return 0 ; 
}






