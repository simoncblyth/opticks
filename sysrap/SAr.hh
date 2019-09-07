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
struct SAr
==============

**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAr
{
    static SAr* Instance ; 

    SAr( int argc_ , char** argv_ , const char* envvar=0, char delim=' ' ) ;

    void args_from_envvar( const char* envvar, char delim );
    void sanitycheck() const ; 

    const char* exepath() const ;
    const char* exename() const ;
    static const char* Basename(const char* path);
    std::string argline() const ;
    const char* cmdline() const ;
    const char* get_arg_after(const char* arg, const char* fallback) const ;
    bool has_arg( const char* arg ) const ; 
    void dump() const ;

    int    _argc ;
    char** _argv ; 

    const char* _cmdline ; 
};



