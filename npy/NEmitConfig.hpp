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

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

/**
NEmitConfig
=============

Canonical m_emitcfg is ctor resident of NEmitPhotonsNPY.


**/

struct NPY_API NEmitConfig 
{
    static const char* DEFAULT ; 

    NEmitConfig(const char* cfg);

    struct BConfig* bconfig ;  
    std::string desc() const  ;
    void dump(const char* msg="NEmitConfig::dump") const ; 

    int verbosity ; 
    int photons ; 
    int wavelength ; 

    float time ; 
    float weight ; 
    float posdelta ;  // nudge photon start position along its direction 

    std::string sheetmask ; 

    float umin ; 
    float umax ; 
    float vmin ; 
    float vmax ; 

    int diffuse ; 
    float ctmindiffuse ;  
    float ctmaxdiffuse ;  




};

