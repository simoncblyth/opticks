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

// TEST=SEnvTest om-t

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "OPTICKS_LOG.hh"

/**
SEnvTest
=========

Dump envvars matching the prefix, or all when no prefix.
Used for checking "fabricated" ctest, ie ctest without its own 
executable, but instead with its own environment. 

**/

int main(int argc, char** argv, char** envp)
{
    OPTICKS_LOG(argc, argv) ;

    const char* prefix = argc > 1 ? argv[1] : NULL ; 

    LOG(debug) << " prefix " << ( prefix ? prefix : "NONE" ); 

    while(*envp)
    {
        if(prefix != NULL && strncmp(*envp, prefix, strlen(prefix)) == 0) 
        { 
            LOG(info) << *envp ;  
        } 
        envp++ ; 
    }

    return 0 ; 
}
