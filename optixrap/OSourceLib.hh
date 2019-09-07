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

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>


class GSourceLib ;
template <typename T> class NPY ;

#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

/**
OSourceLib
===========

Converts the GSourceLib buffer into a texture that 
is accessible in the OptiX context GPU side using::

   source_texture
   source_domain

**/

class OXRAP_API OSourceLib : public OPropertyLib {
    public:
        OSourceLib(optix::Context& ctx, GSourceLib* lib);
    public:
        void convert();
    private:
        void makeSourceTexture(NPY<float>* buf);
    private:
        GSourceLib*    m_lib ;
};







