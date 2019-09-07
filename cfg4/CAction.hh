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

class CFG4_API CAction 
{
    public:
    enum 
    {
       PRE_SAVE         = 0x1 << 0,
       POST_SAVE        = 0x1 << 1,
       PRE_DONE         = 0x1 << 2,
       POST_DONE        = 0x1 << 3,
       LAST_POST        = 0x1 << 4,
       SURF_ABS         = 0x1 << 5,
       PRE_SKIP         = 0x1 << 6,
       MAT_SWAP         = 0x1 << 7,
       STEP_START       = 0x1 << 8,
       STEP_REJOIN      = 0x1 << 9,
       STEP_RECOLL      = 0x1 << 10,
       RECORD_TRUNCATE  = 0x1 << 11,
       BOUNCE_TRUNCATE  = 0x1 << 12,
       ZERO_FLAG        = 0x1 << 13,
       DECREMENT_DENIED = 0x1 << 14,
       HARD_TRUNCATE    = 0x1 << 15,
       TOPSLOT_REWRITE  = 0x1 << 16,
       POST_SKIP        = 0x1 << 17
    };

    static const char* PRE_SAVE_ ; 
    static const char* POST_SAVE_ ; 
    static const char* PRE_DONE_ ; 
    static const char* POST_DONE_ ; 
    static const char* LAST_POST_ ; 
    static const char* SURF_ABS_ ; 
    static const char* PRE_SKIP_ ; 
    static const char* MAT_SWAP_ ; 
    static const char* STEP_START_ ; 
    static const char* STEP_REJOIN_ ; 
    static const char* STEP_RECOLL_ ; 
    static const char* RECORD_TRUNCATE_ ; 
    static const char* BOUNCE_TRUNCATE_ ; 
    static const char* HARD_TRUNCATE_ ; 
    static const char* ZERO_FLAG_ ; 
    static const char* DECREMENT_DENIED_ ; 
    static const char* TOPSLOT_REWRITE_ ; 
    static const char* POST_SKIP_ ; 

    static std::string Action(int action);

};
#include "CFG4_TAIL.hh"

