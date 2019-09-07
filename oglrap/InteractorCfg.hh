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
#include "Interactor.hh"
#include "BCfg.hh"
#include "OGLRAP_API_EXPORT.hh"

template <class Listener>
class OGLRAP_API InteractorCfg : public BCfg {
public:
   InteractorCfg(const char* name, Listener* listener, bool live) : BCfg(name, live) 
   {   
       addOptionF<Listener>(listener, Listener::DRAGFACTOR, "Drag factor");
       addOptionI<Listener>(listener, Listener::OPTIXMODE, "OptiX mode");
   }   
};


template class OGLRAP_API InteractorCfg<Interactor> ;

