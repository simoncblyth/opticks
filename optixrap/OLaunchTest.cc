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

#include <sstream>
#include <iomanip>

#include "OLaunchTest.hh"

// optickscore-
#include "Opticks.hh"

// optixrap-
#include "OContext.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>

using namespace optix ; 

#include "PLOG.hh"

OLaunchTest::OLaunchTest(OContext* ocontext, Opticks* opticks, const char* ptx, const char* prog, const char* exception) 
   :
    m_ocontext(ocontext),
    m_opticks(opticks),
    m_ptx(strdup(ptx)),
    m_prog(strdup(prog)),
    m_exception(strdup(exception)),

    m_entry_index(-1),
    m_width(1),
    m_height(1)
{
    init();
}

void OLaunchTest::setWidth(unsigned int width)
{
    m_width = width ; 
}
void OLaunchTest::setHeight(unsigned int height)
{
    m_height = height ; 
}


void OLaunchTest::init()
{
    m_context = m_ocontext->getContext();
    m_entry_index = m_ocontext->addEntry(m_ptx, m_prog, m_exception);

    LOG(info) << brief() ;  
}


void OLaunchTest::prelaunch()
{
    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  0, 0);
}

void OLaunchTest::launch()
{
    LOG(info) << brief() ;  
    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  m_width, m_height);
}

void OLaunchTest::launch(unsigned int width, unsigned int height)
{
    LOG(info) << brief() ;  
    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  width, height);
}

std::string OLaunchTest::brief()
{  
    std::stringstream ss ; 

    ss << "OLaunchTest"
       << " entry " << std::setw(3) << m_entry_index
       << " width " << std::setw(7) << m_width 
       << " height " << std::setw(7) << m_height
       << " ptx " << std::setw(50) << m_ptx
       << " prog " << std::setw(50) << m_prog
       ;

    return ss.str();
}  



