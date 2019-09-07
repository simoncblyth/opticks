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

#include "X4ThreeVector.hh"

#include "OPTICKS_LOG.hh"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    G4ThreeVector v(1.1,2.2,3.000003 ); 

    std::string c = X4ThreeVector::Code( v, "v" ); 
    LOG(info) << c ; 

    std::string c2 = X4ThreeVector::Code( v, NULL ); 
    LOG(info) << c2 ; 


    return 0 ; 
}

