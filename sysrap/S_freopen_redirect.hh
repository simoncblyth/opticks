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

// https://stackoverflow.com/questions/1908687/how-to-redirect-the-output-back-to-the-screen-after-freopenout-txt-a-stdo

/**
S_freopen_redirect
===================

struct for file descriptor gymnastics. 

Used from :doc:`/optixrap/OContext` to redirect OptiX kernel debug stdout to a file.

**/


#include <cstdio>
#include <cstring>
#include <unistd.h>

struct S_freopen_redirect
{
    FILE*       _old    ;
    int         _old_fd ; 
    const char* _path    ; 

    S_freopen_redirect( FILE* curr, const char* path )
        :
        _old(curr),
        _old_fd(dup(fileno(curr))),
        _path(strdup(path))
    {
        freopen( _path, "w", curr );
    }

    ~S_freopen_redirect()
    {
        fclose(_old);
        FILE *fp2 = fdopen(_old_fd, "w");
        *_old = *fp2;  // Unreliable!
    }                      
};

