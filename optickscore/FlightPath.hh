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

#include <string>
#include <vector>

struct SMeta ; 
class SCtrl ; 

// npy-
template<typename T> class NPY ; 

// opticks-
class InterpolatedView ; 
class Composition ; 

/**
FlightPath
============

Canonical m_flighpath instance is resident of Opticks
and is instanciated by Opticks::getFlightPath

Composition/control hookup is done by OpticksHub::configureFlightPath
which is invoked from OpticksHub::configureVizState

Note that FlightPath can in principal be used in a pure compute 
manner with no OpenGL involvement, eg for making pure raytrace movies
on headless nodes without OpenGL capability.

This is now realised by okop/OpTracer.cc with the OpticksHub::configureFlightPath
being invoked from OpTracer::flightpath

**/

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class SRenderer ; 
class Opticks ; 
struct NFlightConfig ; 

#include "plog/Severity.h"

/**
FlightPath
============

Flightpath is a resident of Opticks that is lazily  
instanciated by Opticks::getFlightPath 

**/

class OKCORE_API FlightPath {
public:
    static const plog::Severity LEVEL ; 

    FlightPath(const Opticks* ok, const char* cfg, const char* outdir, const char* nameprefix);
private:
    void init(); 
public:
    std::string description(const char* msg="FlightPath");
    void Summary(const char* msg="FlightPath::Summary");
public:
    unsigned getNumViews() const ;
    void setCtrl(SCtrl* ctrl); 
public:
    int render(SRenderer* renderer);
public:
    void setVerbose(bool verbose=true);
    void setInterpolatedViewPeriod(unsigned int ivperiod); 
    void setScale(float scale);
    float getScale0() const ;
    float getScale1() const ;
    unsigned getFrameLimit() const ; 
    unsigned getPeriod() const ;

    void refreshInterpolatedView();
    InterpolatedView* getInterpolatedView();
    bool isValid() const  ;
private:
    void load();
    InterpolatedView* makeInterpolatedView();
public:
    int* getIVPeriodPtr();
public:
    void setPathFormat();
    void fillPathFormat(char* path, unsigned path_size, unsigned index );
public:
    void record(double dt ); 
    void getMinMaxAvg(double& mn, double& mx, double& av) const ;
    template<typename T> void setMeta(const char* key, T value); 
    void save() const ; 
private:
    void setPathFormat(const char* path_format);
private:
    const Opticks*                       m_ok ; 
    Composition*                         m_composition ; 
    NFlightConfig*                       m_cfg ; 
    const char*                          m_nameprefix ; 
    const char*                          m_flight ; 
    std::string                          m_inputpath ;
    NPY<float>*                          m_eluc ;  
    InterpolatedView*                    m_view ;  
    bool                                 m_verbose ; 
    int                                  m_ivperiod ; 
    SCtrl*                               m_ctrl ; 
    SMeta*                               m_meta ; 
    float                                m_scale ; 
    const char*                          m_path_format ;  
    std::vector<double>                  m_frame_times ; 
    const char*                          m_outdir ; 

};

#include "OKCORE_TAIL.hh"

