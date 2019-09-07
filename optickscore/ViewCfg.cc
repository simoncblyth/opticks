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
#include "View.hh"
#include "ViewCfg.hh"


template OKCORE_API void BCfg::addOptionF<View>(View*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<View>(View*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<View>(View*, const char*, const char* );





template <class Listener>
ViewCfg<Listener>::ViewCfg(const char* name, Listener* listener, bool live) 
   : 
   BCfg(name, live) 
{
       addOptionS<Listener>(listener, "eye", "Comma delimited eye position in model-extent coordinates, eg 0,0,-1  ");
       addOptionS<Listener>(listener, "look","Comma delimited look position in model-extent coordinates, eg 0,0,0  ");
       addOptionS<Listener>(listener, "up",  "Comma delimited up direction in model-extent frame, eg 0,1,0 " );
}




template class OKCORE_API ViewCfg<View> ;

