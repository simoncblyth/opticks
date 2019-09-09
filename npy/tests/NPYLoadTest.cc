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

// om-;TEST=NPYLoadTest om-t

#include "NPY_FLAGS.hh"


#include <vector>
#include <iostream>
#include <cassert>

#include "SSys.hh"
#include "SStr.hh"

#include "BOpticksEvent.hh"
#include "BBufSpec.hh"

// npy-
#include "NGLM.hpp"
#include "NLoad.hpp"
#include "NPY.hpp"
#include "DummyPhotonsNPY.hpp"

#include "OPTICKS_LOG.hh"


void test_debugload(const char* path, char type)
{
    LOG(info) 
         << "test_debugload " 
         << " path : " << path
         << " type : " << type
         ; 

    NPYBase* npy = NULL ; 
    switch( type )
    {
       case 'f':  npy = NPY<float>::debugload(path); break ; 
       case 'd':  npy = NPY<double>::debugload(path); break ; 
       case 'u':  npy = NPY<unsigned>::debugload(path); break ; 
       case 'i':  npy = NPY<int>::debugload(path); break ; 
       default:   ; break ;
    }
    if(npy) npy->Summary(path);
}


int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    NPYBase::setGlobalVerbose(true);

    const char* path = argc > 1 ? argv[1] : NULL ; 
    char type = argc > 2 ? argv[2][0] : 'u' ; 

    if( path == NULL ) return 0 ; 

    test_debugload( path, type ); 

    return 0 ;
}
