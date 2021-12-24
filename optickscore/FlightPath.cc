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

#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>

#include "SMeta.hh"
#include "SPath.hh"
#include "SRenderer.hh"

// brap-
#include "BFile.hh"
#include "BDir.hh"
#include "PLOG.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NFlightConfig.hpp"

#include "NP.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksConst.hh"
#include "View.hh"
#include "InterpolatedView.hh"
#include "FlightPath.hh"
#include "Composition.hh"



const plog::Severity FlightPath::LEVEL = PLOG::EnvLevel("FlightPath", "DEBUG" ); 

void FlightPath::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}

FlightPath::FlightPath(const Opticks* ok, const char* cfg, const char* outdir, const char* nameprefix)  
    :
    m_ok(ok), 
    m_composition(ok->getComposition()),
    m_cfg(new NFlightConfig(cfg)),
    m_nameprefix(strdup(nameprefix)), 
    m_flight(m_cfg->flight.c_str()),
    m_inputpath(m_ok->getFlightInputPath(m_flight)),
    m_eluc(NULL),
    m_view(NULL),
    m_verbose(false),
    m_ivperiod(128),
    m_ctrl(NULL),
    m_meta(new SMeta),
    m_scale(1.f),
    m_path_format(nullptr),
    m_outdir(strdup(outdir))
{
    init(); 
    LOG(LEVEL) << " m_inputpath " << m_inputpath ; 
}

void FlightPath::init()
{
    setPathFormat(); 
}


void FlightPath::getMinMaxAvg(double& mn, double& mx, double& av) const 
{
    const std::vector<double>& t = m_frame_times ; 

    typedef std::vector<double>::const_iterator IT ;     
    IT mn_ = std::min_element( t.begin(), t.end()  ); 
    IT mx_ = std::max_element( t.begin(), t.end()  ); 
    double sum = std::accumulate(t.begin(), t.end(), 0. );   

    mn = *mn_ ; 
    mx = *mx_ ; 
    av = t.size() > 0 ? sum/double(t.size()) : -1. ;  
}

template<typename T>
void FlightPath::setMeta(const char* key, T value)
{
    nlohmann::json& js = m_meta->js ; 
    js[key] = value ; 
}

void FlightPath::save() const 
{
    nlohmann::json& js = m_meta->js ; 

    js["argline"] = m_ok->getArgLine(); 

    const char* cfg = m_cfg->getCfg() ; 
    const char* emm = m_ok->getEnabledMergedMesh() ; 

    assert( m_nameprefix ) ; // defaults to "frame"

    js["cfg"] = cfg ? cfg : "" ; 
    js["emm"] = emm ? emm : ""  ;  
    js["nameprefix"] = m_nameprefix ;  
    js["scale"] = m_scale ;  

    double mn, mx, av ; 
    getMinMaxAvg(mn, mx, av); 

    js["mn"] = mn ;  
    js["mx"] = mx ;  
    js["av"] = av ;  

    std::string js_name = m_nameprefix ; 
    js_name += ".json" ; 
    m_meta->save(m_outdir,  js_name.c_str() ); 
        
    std::string np_name = m_nameprefix ; 
    np_name += ".npy" ;  

    NP::Write(m_outdir, np_name.c_str(), (double*)m_frame_times.data(),  m_frame_times.size() );  
}

int* FlightPath::getIVPeriodPtr()
{
    return &m_ivperiod ; 
}
unsigned FlightPath::getNumViews() const 
{
    return m_eluc->getNumItems(); 
}
void FlightPath::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
void FlightPath::setInterpolatedViewPeriod(unsigned int ivperiod)
{
    m_ivperiod = ivperiod ; 
}
void FlightPath::setScale(float scale)
{
    m_scale = scale ; 
}

float FlightPath::getScale0() const
{
    return m_cfg->scale0 * m_scale ; 
}
float FlightPath::getScale1() const
{
    return m_cfg->scale1 * m_scale ; 
}
unsigned FlightPath::getFrameLimit() const
{
    return m_cfg->getFrameLimit() ; 
}
unsigned FlightPath::getPeriod() const 
{
    return m_cfg->period ; 
}

void FlightPath::load()
{
    const char* path = m_inputpath.c_str() ; 
    LOG(info) << " path " << path ; 
    delete m_eluc ; 
    m_eluc = NPY<float>::load(path) ; 

    if(m_eluc == NULL)
    {
        LOG(fatal) 
            << " MISSING expected path " << path << " for flight "  << m_flight
            << " (bad name OR need to run ana/makeflight.sh)"
            ;
    }
}

void FlightPath::setPathFormat()
{
    std::string name = m_cfg->getFrameName(m_nameprefix, -1); 
    bool create = true ; 
    std::string fmt = BFile::preparePath(m_outdir ? m_outdir : "$TMP", nullptr, name.c_str(), create);  
    LOG(info) 
        << " m_outdir " << m_outdir 
        << " name " << name
        << " fmt " << fmt 
        ;   

    setPathFormat(fmt.c_str());   
}

void FlightPath::setPathFormat(const char* path_format)
{
    m_path_format = strdup(path_format); 
}

/**
FlightPath::fillPathFormat
---------------------------

Writes to the *path* char array using m_path_format which is expected to 
have one integer format token.

**/

void FlightPath::fillPathFormat(char* path, unsigned path_size, unsigned index )
{
    assert( m_path_format && "must setPathFormat before calling fillPathFormat" ); 
    snprintf(path, path_size, m_path_format, index );
}

void FlightPath::record(double dt)
{
    m_frame_times.push_back(dt); 
}

std::string FlightPath::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << m_inputpath ; 
    return ss.str();
}

void FlightPath::Summary(const char* msg)
{
    LOG(info) << description(msg);
}

bool FlightPath::isValid() const 
{
    return m_eluc != nullptr ; 
}

InterpolatedView* FlightPath::makeInterpolatedView()
{
    load(); 
    return m_eluc ? InterpolatedView::MakeFromArray( m_eluc, m_ivperiod, getScale0(), getScale1(), m_ctrl  ) : nullptr ; 
}


void FlightPath::refreshInterpolatedView()
{
    delete m_view ; 
    m_view = NULL ; 
}

InterpolatedView* FlightPath::getInterpolatedView()
{
    if(!m_view) m_view = makeInterpolatedView();
    return m_view ;             
}


/**
FlightPath::render
--------------------

Invoked for example from OpTracer::render_flightpath

**/

int FlightPath::render( SRenderer* renderer )
{
    LOG(LEVEL) << "[" ; 
 
    FlightPath* fp = this ; 

    int rc = m_composition->setViewType(View::FLIGHTPATH);

    if( rc != 0 )
    {
        LOG(fatal) 
            << "Composition::setViewType FAILED , probably input eye-look-up-ctrl .npy file is missing, see ana/makeflight.py " ;  
            ;
        return 1 ; 
    } 

    InterpolatedView* iv = m_composition->getInterpolatedView() ; 

    if(iv == nullptr)
    {
        LOG(fatal) 
            << "Composition::getInterpolatedView FAILED " ;  
            ;
        return 2 ; 
    }

    unsigned period = fp->getPeriod();     // typical values 4,8,16   (1:fails with many nan)

    iv->setAnimatorModeForPeriod(period);  // change animation "speed" to give *period* interpolation steps between views         

    int framelimit = fp->getFrameLimit(); 

    int total_period = iv->getTotalPeriod(); 

    int imax = framelimit > 0 ? std::min( framelimit, total_period)  : total_period ; 

    std::string top_annotation = m_ok->getContextAnnotation(); 

    unsigned anno_line_height = m_ok->getAnnoLineHeight() ; 

    LOG(LEVEL) 
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


    char path[256] ; 
    for(int i=0 ; i < imax ; i++)
    {
        m_composition->tick();  // changes Composition eye-look-up according to InterpolatedView flightpath

        double dt = renderer->render();   // eg calling OTracer::trace_  
        // Q: where does OTracer pay heed to the changed Composition view position ?
        // A: in OTracer::trace_ where the eye-look-up are accessed from Composition and 
        //    the OptiX context updated accordingly          

        std::string bottom_annotation = m_ok->getFrameAnnotation(i, imax, dt ); 

        fp->fillPathFormat(path, 256, i ); 

        fp->record(dt);  

        LOG(info) << bottom_annotation << " : " << path ; 

        renderer->snap(path, bottom_annotation.c_str(), top_annotation.c_str(), anno_line_height );  // downloads GPU output_buffer pixels into image file
    }

    fp->save(); 
    LOG(LEVEL) << "]" ; 

    return 0 ; 
}



template void FlightPath::setMeta<unsigned long long>(const char*, unsigned long long ) ; 
template void FlightPath::setMeta<unsigned>(const char*, unsigned ) ; 
template void FlightPath::setMeta<int>(const char*, int ) ; 
template void FlightPath::setMeta<float>(const char*, float ) ; 

template void FlightPath::setMeta<std::string>(const char*, std::string ) ; 
template void FlightPath::setMeta<const char*>(const char*, const char* ) ; 
template void FlightPath::setMeta<char*>(const char*,  char* ) ; 

