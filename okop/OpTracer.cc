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

#include "View.hh"
#include "InterpolatedView.hh"
#include "Animator.hh"
#include "FlightPath.hh"
#include "Snap.hh"


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


double OpTracer::render()
{     
    LOG(LEVEL) << "[" ;

    if(m_count == 0 ) setup_render_target(); 

    double dt = m_otracer->trace_();

    m_count++ ; 

    LOG(LEVEL) << "]" ;

    return dt ; 
}   

void OpTracer::snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height)
{
    m_ocontext->snap(path, bottom_line, top_line, line_height ); 
}


/**
OpTracer::render_snap
----------------------

Takes one or more GPU raytrace snapshots of geometry
at various positions configured via --snapconfig
for example::

    --snapconfig="steps=5,eyestartz=0,eyestopz=1"

**/

void OpTracer::render_snap()   
{
    LOG(LEVEL) << "[" ;

    Snap* snap = m_ok->getSnap((SRenderer*)this);
    snap->render(); 

    m_otracer->report("OpTracer::render_snap");   // saves for runresultsdir

    LOG(LEVEL) << "]" ;
}

/**
OpTracer::render_flightpath
-----------------------------

**/

void OpTracer::render_flightpath()   
{
    LOG(LEVEL) << "[" ;

    FlightPath* fp = m_ok->getFlightPath();   // FlightPath lazily instanciated here (held by Opticks)

    //m_hub->setupFlightPathCtrl();     // m_ctrl setup currently only needed for interactive flightpath running ?

    fp->render( (SRenderer*)this  );  

    LOG(LEVEL) << "]" ;
}

