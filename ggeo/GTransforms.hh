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
class GGEO_API GTransforms {
    public:
        static GTransforms* make(unsigned int n);
    public:
        void save(const char* path);
        static GTransforms* load(const char* path);
        NPY<float>* getBuffer();
    public:
        GTransforms(NPY<float>* buf=NULL);
    public:
        void add(const glm::mat4& mat);
        void add();    // identity
    public:
        glm::mat4 get(unsigned int i);
    private:
        NPY<float>* m_buffer ; 

};



