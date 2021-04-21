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
    m_snapoverrideprefix(m_ok->getSnapOverridePrefix()),  // --snapoverrideprefix
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
        << " snapoverrideprefix " << m_snapoverrideprefix
        ;

    int num_steps = m_snap_config->steps ; 

    if( num_steps == 0)
    {
        const char* path = m_snap_config->getSnapPath(dir, reldir, 0, m_snapoverrideprefix ); 
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
    std::vector<glm::vec3>    eyes ; 
    m_composition->eye_sequence(eyes, m_snap_config ); 
    const char* path_fmt = m_snap_config->getSnapPath(dir, reldir, -1, m_snapoverrideprefix); 
    multi_snap(path_fmt, eyes ); 
}

void OpTracer::multi_snap(const char* path_fmt, const std::vector<glm::vec3>& eyes )
{
    char path[128] ; 
    for(int i=0 ; i < int(eyes.size()) ; i++)
    {
        const glm::vec3& eye = eyes[i] ; 
        m_composition->setEye( eye.x, eye.y, eye.z ); 

        snprintf(path, 128, path_fmt, i );   
        single_snap(path);  
    }
}



/**
OpTracer::single_snap
------------------------

Single snap uses composition targetting 

**/

void OpTracer::single_snap(const char* path)
{
    double dt = render();   // OTracer::trace_

    std::string top_annotation = m_ok->getContextAnnotation(); 
    std::string bottom_annotation = m_ok->getFrameAnnotation(0, 1, dt ); 
    unsigned anno_line_height = m_ok->getAnnoLineHeight() ; 

    LOG(info) << top_annotation << " | " << bottom_annotation << " | " << path << " | " << anno_line_height ; 

    m_ocontext->snap(path, bottom_annotation.c_str(), top_annotation.c_str(), anno_line_height );
} 



/**

Hmm its better now, but still most of this belongs elsewhere. 
Its kinda involved as have to coordinate between several objects.

Hmm: would be better to use some protocols so the OpTracer instance 
can can be effectively passed down to methods that do the below at the lower Opticks level 
and call back up to OpTracer level to *render* and *snap* 

Then can eliminate this method::

    m_ok->flightpath(this) ; 

**/

void OpTracer::flightpath(const char* dir, const char* reldir )   
{
    m_hub->setupFlightPath();   // FlightPath instanciated here and held by Opticks

    m_composition->setViewType(View::FLIGHTPATH);

    InterpolatedView* iv = m_composition->getInterpolatedView() ; assert(iv); 

    FlightPath* fp = m_ok->getFlightPath(); 

    fp->setPathFormat(dir, reldir); 

    unsigned period = fp->getPeriod();     // typical values 4,8,16   (1:fails with many nan)

    iv->setAnimatorModeForPeriod(period);  // change animation "speed" to give *period* interpolation steps between views         

    int framelimit = fp->getFrameLimit(); 

    int total_period = iv->getTotalPeriod(); 

    int imax = framelimit > 0 ? std::min( framelimit, total_period)  : total_period ; 

    std::string top_annotation = m_ok->getContextAnnotation(); 

    unsigned anno_line_height = m_ok->getAnnoLineHeight() ; 

    LOG(info) 
        << " period " << period
        << " total_period " << total_period
        << " framelimit " << framelimit << " (OPTICKS_FLIGHT_FRAMELIMIT) " 
        << " imax " << imax 
        << " top_annotation " << top_annotation
        << " anno_line_height " << anno_line_height
        ;

    fp->setMeta<int>("framelimit", framelimit); 
    fp->setMeta<int>("total_period", total_period); 
    fp->setMeta<int>("imax", imax); 
    fp->setMeta<std::string>("top_annotation", top_annotation ); 


    char path[128] ; 
    for(int i=0 ; i < imax ; i++)
    {
        m_composition->tick();  // changes Composition eye-look-up according to InterpolatedView flightpath

        double dt = render();   // calling OTracer::trace_
        
        std::string bottom_annotation = m_ok->getFrameAnnotation(i, imax, dt ); 

        fp->fillPathFormat(path, 128, i ); 

        fp->record(dt);  

        LOG(info) << bottom_annotation << " path " << path ; 

        m_ocontext->snap(path, bottom_annotation.c_str(), top_annotation.c_str(), anno_line_height );  // downloads GPU output_buffer pixels into image file
    }

    fp->save(); 

    LOG(info) << "]" ;
}

