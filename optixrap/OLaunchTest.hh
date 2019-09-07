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
OLaunchTest
============

OptiX program launcher

**/


#include "OXPPNS.hh"
#include <string>
class Opticks ; 
class OContext ; 

#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OLaunchTest {
    public:
        OLaunchTest(OContext* ocontext, Opticks* opticks, const char* ptx="textureTest.cu.ptx", const char* prog="textureTest", const char* exception="exception"); 
        std::string brief();
    public:
        void setWidth(unsigned int width);
        void setHeight(unsigned int height);
        void prelaunch();
        void launch();
        void launch(unsigned width, unsigned height);
    private:
        void init();
    private:
        OContext*        m_ocontext ; 
        Opticks*         m_opticks ; 
        optix::Context   m_context ;

        const char*      m_ptx ; 
        const char*      m_prog ; 
        const char*      m_exception ; 

        int              m_entry_index ; 
        unsigned int     m_width ; 
        unsigned int     m_height ; 
 

};


