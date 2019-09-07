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
OAxisTest
===========

For standalone testing of OptiX and OpenGL integration.

**/


class OContext ; 
template <typename T> class NPY ; 

#include "OXRAP_PUSH.hh"
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include "OKGL_API_EXPORT.hh"
#include "SLauncher.hh"

class OKGL_API OAxisTest : public SLauncher {
    public:
        OAxisTest(OContext* ocontext, NPY<float>* axis_data);
        void prelaunch();
        void download();
    public:
        virtual void launch(unsigned count);
    private:
        void init();
    private:
        OContext*       m_ocontext ;
        NPY<float>*     m_axis_data ; 
        optix::Buffer   m_buffer ; 
        unsigned        m_ni ; 
        unsigned        m_entry ; 
};


