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

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CStage 
{
    public:
    typedef enum { UNKNOWN, START, COLLECT, REJOIN, RECOLL } CStage_t ;
    static const char* UNKNOWN_ ;
    static const char* START_  ;
    static const char* COLLECT_  ;
    static const char* REJOIN_  ;
    static const char* RECOLL_  ;
    static const char* Label( CStage_t stage);
};

#include "CFG4_TAIL.hh"

