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
#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

/**
SId
====

Supply single char identifiers from ctor argument string, 
until run out at which point cycle is incremented so give
an integer suffix. 

reset returns to the first identifier.

This is used for code generation in X4Solid, search for g4code.
 
**/


struct SYSRAP_API SId 
{
    SId(const char* identifiers_);  

    const char* get(bool reset=false); 
    void reset(); 

    const char* identifiers ; 
    int         len ; 
    int         idx ; 
    int         cycle ; 
};


