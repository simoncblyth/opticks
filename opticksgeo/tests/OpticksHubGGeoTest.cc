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


#include "NLookup.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GGeoTest.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    //const char* funcname = "tboolean-torus--" ;
    const char* funcname = "tboolean-media--" ;
    //const char* funcname = "tboolean-nonexisting--" ;

    Opticks ok(argc, argv, GGeoTest::MakeArgForce(funcname, "--dbgsurf --dbgbnd") );

    OpticksHub hub(&ok);      

    if(hub.getErr()) LOG(fatal) << "hub error " << hub.getErr() ; 

    // NLookup* lookup = hub.getLookup();
    // lookup->close();  


    return 0 ; 
}
