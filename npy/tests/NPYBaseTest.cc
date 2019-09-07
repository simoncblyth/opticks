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

// TEST=NPYBaseTest om-t 

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;


    int acv = 1001 ; 

    NPY<int>* np = NPY<int>::make(1,1,4) ; 
    np->setArrayContentVersion(acv); 

    const char* path="$TMP/npy/NPYBaseTest/acv.npy" ; 
    np->save(path); 

    NPY<int>* np2 = NPY<int>::load( path ) ; 
    int acv2 = np2->getArrayContentVersion(); 

    LOG(info) << " acv2 " << acv2 ; 

    assert( acv2 == acv ); 



    return 0 ; 
}
