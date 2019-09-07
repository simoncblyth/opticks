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

/**
SLog
======

Trivial logger enabling bracketing 
of constructor initializer lists.

**/


#include "SYSRAP_API_EXPORT.hh"
#include "plog/Severity.h"
class SYSRAP_API SLog 
{
    public:
        static const char* exename();
        static void Nonce(); 
    public:
        SLog(const char* label, const char* extra="", plog::Severity=plog::info );
        void operator()(const char* msg="");
    private:
        const char* m_label ; 
        const char* m_extra ; 
        plog::Severity m_level ; 
};


