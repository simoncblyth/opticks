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

#include "SSys.hh"
#include "SPPM.hh"
#include "BFile.hh"
#include "PLOG.hh"
// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSnapConfig.hpp"

// okc-
#include "Opticks.hh"
#include "Composition.hh"


// okg-
#include "OpticksHub.hh"

// optixrap-
#include "OContext.hh"
#include "OTracer.hh"

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

// opop-
#include "OpEngine.hh"
#include "OpTracer.hh"

const plog::Severity OpTracer::LEVEL = PLOG::EnvLevel("OpTracer", "DEBUG") ; 

int OpTracer::Preinit()  // static
{
    LOG(LEVEL); 
    return 0 ; 
}

OpTracer::OpTracer(OpEngine* ope, OpticksHub* hub, bool immediate) 
    :
    m_preinit(Preinit()), 
    m_ope(ope),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_snap_config(m_ok->getSnapConfig()),
    m_immediate(immediate),

    m_ocontext(NULL),   // defer 
    m_composition(m_hub->getComposition()),
    m_otracer(NULL),
    m_count(0)
{
    init();
}

void OpTracer::init()
{
    LOG(LEVEL); 
    if(m_immediate)
    {
        initTracer();
    }
}


void OpTracer::initTracer()
{
    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    LOG(LEVEL)
        << "["
        << " width " << width 
        << " height " << height 
        << " immediate " << m_immediate
        ;

    m_ocontext = m_ope->getOContext();

    optix::Context context = m_ocontext->getContext();

    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

    context["output_buffer"]->set( output_buffer );

    m_otracer = new OTracer(m_ocontext, m_composition);

    LOG(LEVEL) << "]" ;
}


void OpTracer::setup_render_target()
{
    LOG(LEVEL) << "[" ;
    m_hub->setupCompositionTargetting();
    m_otracer->setResolutionScale(1) ;
    LOG(LEVEL) << "]" ;
}


void OpTracer::render()
{     
    LOG(LEVEL) << "[" ;
    if(m_count == 0 ) setup_render_target(); 
    m_otracer->trace_();
    m_count++ ; 
    LOG(LEVEL) << "]" ;
}   


/**
OpTracer::snap
----------------

Takes one or more GPU raytrace snapshots of geometry
at various positions configured via --snapconfig
for example::

    --snapconfig="steps=5,eyestartz=0,eyestopz=1"

**/

void OpTracer::snap(const char* dir, const char* reldir)   
{
    LOG(info) 
        << "[" << m_snap_config->desc()
        << " dir " << dir 
        << " reldir " << reldir 
        ;

    int num_steps = m_snap_config->steps ; 

    if( num_steps == 0)
    {
        const char* path = m_snap_config->getSnapPath(dir, reldir, -1); 
        single_snap(path);  
    }
    else
    {
        multi_snap(dir, reldir); 
    }

    m_otracer->report("OpTracer::snap");   // saves for runresultsdir

    if(!m_ok->isProduction())
    {
        m_ok->saveParameters(); 
    }

    LOG(info) << "]" ;
}



void OpTracer::multi_snap(const char* dir, const char* reldir)
{
    int num_steps = m_snap_config->steps ; 

    float eyestartx = m_snap_config->eyestartx ; 
    float eyestarty = m_snap_config->eyestarty ; 
    float eyestartz = m_snap_config->eyestartz ; 

    float eyestopx = m_snap_config->eyestopx ; 
    float eyestopy = m_snap_config->eyestopy ; 
    float eyestopz = m_snap_config->eyestopz ; 

    for(int i=0 ; i < num_steps ; i++)
    {
        float frac = num_steps > 1 ? float(i)/float(num_steps-1) : 0.f ; 

        float eyex = m_composition->getEyeX();
        float eyey = m_composition->getEyeY();
        float eyez = m_composition->getEyeZ();

        if(!SSys::IsNegativeZero(eyestartx))
        { 
            eyex = eyestartx + (eyestopx-eyestartx)*frac ; 
            m_composition->setEyeX( eyex ); 
        }
        if(!SSys::IsNegativeZero(eyestarty))
        { 
            eyey = eyestarty + (eyestopy-eyestarty)*frac ; 
            m_composition->setEyeY( eyey ); 
        }
        if(!SSys::IsNegativeZero(eyestartz))
        { 
            eyez = eyestartz + (eyestopz-eyestartz)*frac ; 
            m_composition->setEyeZ( eyez ); 
        }

        const char* path = m_snap_config->getSnapPath(dir, reldir, i); 
        single_snap(path);  
    }
}



void OpTracer::single_snap(const char* path)
{
    
    float eyex = m_composition->getEyeX();
    float eyey = m_composition->getEyeY();
    float eyez = m_composition->getEyeZ();

    std::cout
        << " count " << std::setw(5) << m_count 
        << " eyex " << std::setw(10) << eyex
        << " eyey " << std::setw(10) << eyey
        << " eyez " << std::setw(10) << eyez
        << " path " << path 
        << std::endl ;         

    render();

    m_ocontext->snap(path);
} 

