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

// TEST=OpticksGenstepTest om-t

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "BOpticksResource.hh"
#include "OpticksGenstep.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    LOG(info) << "OpticksGenstep::Dump()" ;
    LOG(info) << std::endl << OpticksGenstep::Dump() ;


    //const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ; 
    const char* def = "$DATADIR/gensteps/dayabay/natural/1.npy" ; 
    const char* path = argc > 1 ? argv[1] : def ; 

    bool testgeo(false); 
    BOpticksResource bor(testgeo) ;  // needed ro resolve internal "envvar" DATADIR, see BResourceTest, BFile 

    NPY<float>* np = NPY<float>::load(path) ; 
    if(np == NULL) return 0 ; 

    OpticksGenstep* gs = new OpticksGenstep(np) ; 

    unsigned modulo = 1000 ; 
    unsigned margin = 10 ;  
    gs->dump( modulo, margin ) ; 

    return 0 ; 
}
