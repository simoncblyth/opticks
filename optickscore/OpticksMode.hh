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
#include <string>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksMode
===============

Constructor resident of Opticks instanciated very early prior to configuration.

**/

class Opticks ; 


class OKCORE_API OpticksMode {
    public:
       static const char* COMPUTE_ARG_ ; 
       static const char* INTEROP_ARG_ ; 
       static const char* NOVIZ_ARG_ ; 
    public:
       static const char* UNSET_MODE_ ;
       static const char* COMPUTE_MODE_ ;
       static const char* INTEROP_MODE_ ;
       static const char* CFG4_MODE_ ;
       enum {
                UNSET_MODE   = 0x1 << 0, 
                COMPUTE_MODE = 0x1 << 1, 
                INTEROP_MODE = 0x1 << 2, 
                CFG4_MODE    = 0x1 << 3
            }; 
    public:
       static unsigned Parse(const char* tag);
    public:
        OpticksMode(const char* tag);  // used to instanciate from OpticksEvent metadata
        OpticksMode(Opticks* ok);
    public:
        int getInteractivityLevel() const ;
        std::string description() const ;
        bool isCompute() const ;
        bool isInterop() const ;
        bool isCfG4() const ;   // needs manual override to set to CFG4_MODE
    public:
        void setOverride(unsigned mode);
    private:
        unsigned  m_mode ;  
        bool      m_compute_requested ;  
        bool      m_noviz ; 
        bool      m_forced_compute ;  
};

#include "OKCORE_TAIL.hh"

