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
#include "Demo.hh"
#include "DemoCfg.hh"


template OKCORE_API void BCfg::addOptionF<Demo>(Demo*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Demo>(Demo*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Demo>(Demo*, const char*, const char* );






template <class Listener>
DemoCfg<Listener>::DemoCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
      addOptionF<Listener>(listener, Listener::A, "A");
      addOptionF<Listener>(listener, Listener::B, "B");
      addOptionF<Listener>(listener, Listener::C, "C");
} 



template class OKCORE_API DemoCfg<Demo> ;


