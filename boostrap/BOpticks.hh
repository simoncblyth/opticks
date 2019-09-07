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

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

struct SArgs ; 
class BOpticksKey ; 
class BOpticksResource ; 


/**
BOpticks
==========

Used as a lower level standin for Opticks
primarily in NPY tests that need to load resources, 
for example npy/tests/NCSG2Test.cc 


**/

class BRAP_API  BOpticks {
    public:
        BOpticks(int argc=0, char** argv=nullptr, const char* argforced=nullptr ); 
    public:
        const char* getPath(const char* rela=nullptr, const char* relb=nullptr, const char* relc=nullptr ) const ; 
        int         getError() const ; 
 
        const char* getFirstArg(const char* fallback=nullptr ) const ; 
        const char* getArg( int n=1, const char* fallback=nullptr) const ; 

    private:
        const char*          m_firstarg ; 
        SArgs*               m_sargs ; 
        int                  m_argc ; 
        char**               m_argv ; 
        bool                 m_envkey ; 
        bool                 m_testgeo ; 
        BOpticksResource*    m_resource ; 
        int                  m_error ; 
       
 
};

#include "BRAP_TAIL.hh"

