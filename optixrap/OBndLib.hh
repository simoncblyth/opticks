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

class GBndLib ; 
class Opticks ; 
template <typename T> class NPY ;

#include "plog/Severity.h"
#include "OPropertyLib.hh"
#include "OXRAP_API_EXPORT.hh"

/**
OBndLib
=========

Translates and uploads into OptiX GPU context:

1. GBndLib NPY buffer into OptiX boundary_texture 
2. GBndLib NPY optical buffer into OptiX optical_buffer 

**/


class OXRAP_API OBndLib  : public OPropertyLib 
{
public:
    static const plog::Severity LEVEL ; 
public:
    OBndLib(optix::Context& ctx, GBndLib* lib);
public:
    unsigned getNumBnd();
    GBndLib* getBndLib(); 

    void setDebugBuffer(NPY<float>* npy);

    void setWidth(unsigned int width);
    void setHeight(unsigned int height);
    unsigned getWidth();
    unsigned getHeight();

    void convert();
private:
    void makeBoundaryTexture(NPY<float>* buf);
    void makeBoundaryOptical(NPY<unsigned int>* obuf);
private:
    GBndLib*             m_blib ; 
    Opticks*             m_ok ; 
    NPY<float>*          m_debug_buffer ; 
    unsigned int         m_width ; 
    unsigned int         m_height ; 


};


