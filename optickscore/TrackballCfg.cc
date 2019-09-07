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
#include "Trackball.hh"
#include "TrackballCfg.hh"

template OKCORE_API void BCfg::addOptionF<Trackball>(Trackball*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Trackball>(Trackball*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Trackball>(Trackball*, const char*, const char* );




template <class Listener>
TrackballCfg<Listener>::TrackballCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
       addOptionF<Listener>(listener, Listener::RADIUS,          "Trackball radius");
       addOptionF<Listener>(listener, Listener::TRANSLATEFACTOR, "Translation factor");

       addOptionS<Listener>(listener, Listener::ORIENTATION,     "Comma delimited theta,phi in degress");
       addOptionS<Listener>(listener, Listener::TRANSLATE,       "Comma delimited x,y,z translation triplet");
}





template class OKCORE_API TrackballCfg<Trackball> ;

