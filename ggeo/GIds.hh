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

#include <glm/fwd.hpp>
template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"
class GGEO_API GIds {
    public:
        static GIds* make(unsigned int n);
    public:
        static GIds* load(const char* path);
        void save(const char* path);
        NPY<unsigned int>* getBuffer();
    public:
        GIds(NPY<unsigned int>* buf=NULL);
    public:
        void add(const glm::uvec4& v);
        void add(unsigned int x, unsigned int y, unsigned int z, unsigned int w); 
    public:
        glm::uvec4 get(unsigned int i);

    private:
        NPY<unsigned int>*  m_buffer; 

};


