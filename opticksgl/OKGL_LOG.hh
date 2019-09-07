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
#include "OKGL_API_EXPORT.hh"

#define OKGL_LOG__  {     OKGL_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKGL"), plog::get(), NULL );  } 

#define OKGL_LOG_ {     OKGL_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKGL_API OKGL_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

