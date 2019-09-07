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

#pragma once

#include <cstdio>
#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

struct BRAP_API BBufSpec {

    int id ; 
    void* ptr ;
    unsigned int num_bytes ; 
    int target ; 

    BBufSpec(int id_, void* ptr_, unsigned int num_bytes_, int target_)
       :
          id(id_),
          ptr(ptr_),
          num_bytes(num_bytes_),
          target(target_)
    {
    }
    void Summary(const char* msg)
    {
        printf("%s : id %d ptr %p num_bytes %d target %d \n", msg, id, ptr, num_bytes, target ); 
    }

};

#include "BRAP_TAIL.hh"

