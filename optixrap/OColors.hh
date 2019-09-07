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
OColors
==========

OptiX GPU side color samplers.

**/

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>
#include "plog/Severity.h"

class OpticksColors ; 
template <typename T> class NPY ; 


#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OColors 
{
public:
    static const plog::Severity LEVEL ;  
public:
    OColors(optix::Context& ctx, OpticksColors* colors);
public:
    void convert();
private:
#ifdef OLD_WAY
    optix::TextureSampler makeColorSampler(NPY<unsigned char>* colorBuffer);
    optix::TextureSampler makeSampler(NPY<unsigned char>* buffer, RTformat format, unsigned int nx, unsigned int ny);
#endif
private:
    optix::Context       m_context ; 
    OpticksColors*       m_colors ; 

};


