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


#include "NGLM.hpp"
#include "Camera.hh"
#include "CameraCfg.hh"


template OKCORE_API void BCfg::addOptionF<Camera>(Camera*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Camera>(Camera*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Camera>(Camera*, const char*, const char* );


template <class Listener>
CameraCfg<Listener>::CameraCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
    addOptionI<Listener>(listener, Listener::PRINT, "Print");

    addOptionF<Listener>(listener, Listener::NEAR_, "Near distance");
    addOptionF<Listener>(listener, Listener::FAR_,  "Far distance" );
    addOptionF<Listener>(listener, Listener::ZOOM,  "Zoom factor");
    addOptionF<Listener>(listener, Listener::SCALE, "Screen Scale, CAUTION this is treated as an input only for orthographic camera type");

    addOptionF<Listener>(listener, Listener::TYPE,  "Perspective/Orthographic/Equirectangular");
    // huh: why F for this ? 
}


template class OKCORE_API CameraCfg<Camera> ;
