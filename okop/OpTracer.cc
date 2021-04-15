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
    m_count(0),
    m_flightpath_snaplimit(SSys::getenvint("OPTICKS_FLIGHTPATH_SNAPLIMIT",3))
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
    std::vector<glm::vec3>    eyes ; 
    m_composition->eye_sequence(eyes, m_snap_config ); 
    const char* path_fmt = m_snap_config->getSnapPath(dir, reldir, -1); 
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






const char* OpTracer::FLIGHTPATH_SNAP = "FlightPath%0.5d.jpg" ; 

void OpTracer::flightpath(const char* dir, const char* reldir )   
{
    bool create = true ; 
    std::string fmt = BFile::preparePath(dir ? dir : "$TMP", reldir, FLIGHTPATH_SNAP, create);  


    LOG(info) 
        << " dir " << dir 
        << " reldir " << reldir 
        << " fmt " << fmt 
        ;

    m_hub->configureFlightPath(); 
    m_composition->setViewType(View::FLIGHTPATH);

    View* view = m_composition->getView(); 

    InterpolatedView* iv = reinterpret_cast<InterpolatedView*>(view); 
    assert(iv); 
    iv->commandMode("TB") ;  // FAST16
    //iv->commandMode("TC") ;  // FAST32
    // iv->commandMode("TD") ;  // FAST64  loadsa elu nan

    unsigned num_views = iv->getNumViews();  
    Animator* anim = iv->getAnimator(); 
    unsigned period = anim->getPeriod();  
    unsigned tot_period = period*num_views ;

    unsigned count(0); 
    char path[128] ; 

    unsigned i1 = m_flightpath_snaplimit > 0 ? std::min( m_flightpath_snaplimit, tot_period)  : tot_period ; 

    LOG(info) 
        << " num_views " << num_views
        << " animator.period " << period
        << " tot_period " << tot_period
        << " m_flightpath_snaplimit " << m_flightpath_snaplimit << " (OPTICKS_FLIGHTPATH_SNAPLIMIT) " 
        << " i1 " << i1 
        ;

    for(unsigned i=0 ; i < i1 ; i++)
    {
        count = m_composition->tick();
        render(); 
        snprintf(path, 128, fmt.c_str(), i );   
        std::cout 
            << "OpTracer::flightpath " 
            << " count " <<  std::setw(6) << count 
            << " i " <<  std::setw(6) << i
            << " path " << path 
            << std::endl 
            ;
        m_ocontext->snap(path);
    }

    LOG(info) << "]" ;
}





