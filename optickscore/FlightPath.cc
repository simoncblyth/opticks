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

// brap-
#include "BFile.hh"
#include "BDir.hh"
#include "PLOG.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NFlightConfig.hpp"

// optickscore-
#include "OpticksConst.hh"
#include "InterpolatedView.hh"
#include "FlightPath.hh"


const char* FlightPath::FILENAME = "flightpath.npy"  ; 

const plog::Severity FlightPath::LEVEL = PLOG::EnvLevel("FlightPath", "DEBUG" ); 

void FlightPath::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}

FlightPath::FlightPath(const char* cfg)  
    :
    m_cfg(new NFlightConfig(cfg)),
    m_flightpathdir(m_cfg->idir.c_str()),
    m_eluc(NULL),
    m_view(NULL),
    m_verbose(false),
    m_ivperiod(128),
    m_ctrl(NULL),
    m_scale(1.f)
{
    LOG(LEVEL) << " m_flightpathdir " << m_flightpathdir ; 
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
    return m_cfg->framelimit ; 
}


void FlightPath::load()
{
    std::string path = BFile::FormPath(m_flightpathdir, FILENAME ) ;
    LOG(info) << " path " << path ; 
    delete m_eluc ; 
    m_eluc = NPY<float>::load(path.c_str()) ; 
    assert( m_eluc ) ; 
}


std::string FlightPath::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " << ( m_flightpathdir ? m_flightpathdir : "NULL" )  ; 
    return ss.str();
}

void FlightPath::Summary(const char* msg)
{
    LOG(info) << description(msg);
}

InterpolatedView* FlightPath::makeInterpolatedView()
{
    load(); 
    assert( m_eluc ) ; 
    return InterpolatedView::MakeFromArray( m_eluc, m_ivperiod, getScale0(), getScale1(), m_ctrl  ) ; 
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


