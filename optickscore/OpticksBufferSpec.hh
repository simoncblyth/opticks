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

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksBufferSpec::Get 
=======================

The static Get method returns a comma delimited string containing buffer 
handling tags for each named buffer which are often different in compute and interop modes. 

Canonical usage is from OpticksEvent::createSpec which sets up the NPYSpec for 
all named event buffers using the buffer spec string as an argument which is 
subsequently accessible with NPYSpec::getCtrl.

The strings are parsed into an enumerated bitfield by okc/OpticksBufferControl.cc

**/

class OKCORE_API OpticksBufferSpec 
{
    public:
        static const char* Get(const char* name, bool compute );
    public:
        static const char*  genstep_compute_ ; 
        static const char*  nopstep_compute_ ; 
        static const char*  photon_compute_ ; 
        static const char*  debug_compute_ ; 
        static const char*  way_compute_ ; 
        static const char*  source_compute_ ; 
        static const char*  record_compute_ ; 
        static const char*  phosel_compute_ ; 
        static const char*  recsel_compute_ ; 
        static const char*  sequence_compute_ ; 
        static const char*  seed_compute_ ; 
        static const char*  hit_compute_ ; 

        static const char*  genstep_interop_ ; 
        static const char*  nopstep_interop_ ; 
        static const char*  photon_interop_ ; 
        static const char*  debug_interop_ ; 
        static const char*  way_interop_ ; 
        static const char*  source_interop_ ; 
        static const char*  record_interop_ ; 
        static const char*  phosel_interop_ ; 
        static const char*  recsel_interop_ ; 
        static const char*  sequence_interop_ ; 
        static const char*  seed_interop_ ; 
        static const char*  hit_interop_ ; 
};

#include "OKCORE_TAIL.hh"


