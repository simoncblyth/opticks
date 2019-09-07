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

//  externals/implicitmesher/implicitmesher/tests/ImplicitMesherFTest.cc

#include <iostream>

#include "ImplicitMesherF.h"
#include "SphereFunc.h"

int main()
{
    sphere_functor sf(0,0,0,10, false ) ; 
    //std::cout << sf.desc() << std::endl ; 

    int verbosity = 3 ; 
    ImplicitMesherF mesher(sf, verbosity, 0.f, false); 

    glm::vec3 min(-20,-20,-20);
    glm::vec3 max( 20, 20, 20);

    mesher.setParam(100, min, max);

    mesher.addSeed( 0,0,0, 1,0,0 );


    mesher.polygonize();
    if(verbosity > 0) mesher.dump();


    return 0 ; 
}

/**

Contrast the approaches:

ImplicitMesherF
    std::function ctor argument, use requires single header only

ImplicitMesher
   templated functor class, requires a boatload of public headers 



**/
 
