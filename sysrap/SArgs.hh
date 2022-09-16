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

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cstring>

#include "plog/Severity.h"

/**
struct SArgs
==============

Allows combining standard arguments with arguments 
from a split string.

TODO: Why is this implemented in the header ? Was is for some logging reason ?
      Investigate and fix as it forces recompilation of  everything following changes.


**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SArgs
{
    static const plog::Severity LEVEL ; 

    int argc ;
    char** argv ; 

    std::vector<std::string> elem ; 
    void add(int argc_, char** argv_);

    static bool starts_with( const std::string& e, const char* pfx );
    void addElements(const std::string& line, bool dedupe);
    void make();

    std::string desc() const ; 
    void dump() const ;

    SArgs(int argc_, char** argv_, const char* argforced, const char* opts, bool dedupe=true);
    SArgs(const char* argv0, const char* argline);
    bool hasArg(const char* arg) const ;
    std::string getArgLine() const ; 
    const char* get_arg_after(const char* option, const char* fallback) const ; 
    const char* get_first_arg_ending_with(const char* ending, const char* fallback) const ; 

};

