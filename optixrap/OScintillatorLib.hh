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

/**
OScintillatorLib
===================

Used to upload GScintillatorLib reemission data
into GPU reemission texture.  


**/


#include "OXPPNS.hh"
#include "plog/Severity.h"
#include <optixu/optixu_math_namespace.h>

class GScintillatorLib ;
template <typename T> class NPY ;

#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OScintillatorLib : public OPropertyLib {
    public:
        static const plog::Severity LEVEL ; 
    public:
        OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib);
    public:
        void convert(const char* slice=NULL);
    private:
        void makeReemissionTexture(NPY<float>* buf);
    private:
        GScintillatorLib*    m_lib ;
        NPY<float>*          m_placeholder ; 
};






