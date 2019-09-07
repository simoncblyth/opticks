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

class Opticks ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

//
// skeleton to hold high level Opticks instances
// intended to take over from the organic ggv-/App
// in a more modular way with helper classes for 
// such things as index presentation prep
//

class OKCORE_API OpticksApp {
   public:
       OpticksApp(Opticks* opticks);
   private:
       Opticks* m_opticks ; 
};

#include "OKCORE_TAIL.hh"



