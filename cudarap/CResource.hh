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

#include "CBufSpec.hh"

struct CResourceImp ; 

/**
CResource
============

Provides OpenGL buffer as a CUDA Resource 

Used from okop : OpIndexer, OpZeroer, OpSeeder


**/


#include "CUDARAP_API_EXPORT.hh"
class CUDARAP_API CResource {
    public:
        typedef enum { RW, R, W } Access_t ; 
    public:
        CResource( unsigned buffer_id, Access_t access );
    private:
        void init(); 
    public:
        template <typename T> CBufSpec mapGLToCUDA();
        void unmapGLToCUDA();
    public:
        void streamSync();
    private:
        CResourceImp*  m_imp ; 
        unsigned int   m_buffer_id ; 
        Access_t       m_access ; 
        bool           m_mapped ; 
};



