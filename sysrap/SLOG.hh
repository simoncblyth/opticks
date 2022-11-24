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
SLOG.hh
==========

See SLOG.rst for notes. 

**/


#include <cstddef>
#include <string>
#include <plog/Log.h>

// NB DANGEROUS "USING" OF LOGLEVEL NAMES
//    INCLUDE THIS AS THE LAST HEADER
//    TO AVOID REPLACING STRINGS IN SYSTEM HEADERS
//
// plog log levels 
using plog::none  ;    // 0
using plog::fatal ;    // 1
using plog::error ;    // 2
using plog::warning ;  // 3 
using plog::info ;     // 4
using plog::debug ;    // 5
using plog::verbose ;  // 6 

#include "SYSRAP_API_EXPORT.hh"

struct SLOG ; 

#include <string>
#include "SAr.hh"

struct STTF ; 

struct SYSRAP_API SLOG 
{
    static const int MAXARGC ; 

    SAr         args ; 
    STTF*       ttf ;    // truetypefont

    int         level ; 
    const char* filename ; 
    int         maxFileSize ;    // bytes
    int         maxFiles ; 


    template<int IDX>
    static plog::Severity MaxSeverity(plog::Logger<IDX>* logger) ; 

    template<int IDX>
    static const char* MaxSeverityString(plog::Logger<IDX>* logger) ; 

    template<int IDX>
    static std::string Desc(plog::Logger<IDX>* logger); 

    template<int IDX>
    static std::string Desc(); 

    static void Dump(); 
    static std::string Flags(); 


    static plog::Severity Delta(plog::Severity level_, int delta); 
    static plog::Severity EnvLevel( const char* key, const char* fallback); 

    SLOG(const char* name, const char* fallback="VERBOSE", const char* prefix=NULL );
    SLOG(int argc, char** argv, const char* fallback="VERBOSE", const char* prefix=NULL );
    void init(const char* fallback, const char* prefix); 
    std::string desc() const ; 

    const char* name(); 
    const char* exename() const ;
    const char* cmdline() const ;
    const char* get_arg_after(const char* option, const char* fallback) const ;
    int         get_int_after(const char* option, const char* fallback) const ;
    bool        has_arg(const char* arg) const ; 

    int parse( const char* fallback);
    int parse( plog::Severity _fallback);

    int prefixlevel_parse( int fallback, const char* prefix);
    int prefixlevel_parse( const char* fallback, const char* prefix);
    int prefixlevel_parse( plog::Severity _fallback, const char* prefix);

    static int  _parse(int argc, char** argv, const char* fallback);
    static int  _prefixlevel_parse(int argc, char** argv, const char* fallback, const char* prefix);
    static void _dump(const char* msg, int argc, char** argv);
    static const char* _name(plog::Severity severity);
    static const char* _name(int level);
    static const char* _logpath_parse_problematic(int argc, char** argv);
    static const char* _logpath();

    static SLOG* instance ; 
};


#include "SLOG_INIT.hh"

// newer plog has _ID  older does not 
#ifdef PLOG_DEFAULT_INSTANCE_ID
#define SLOG_DEFAULT_INSTANCE_ID PLOG_DEFAULT_INSTANCE_ID
#else
#define SLOG_DEFAULT_INSTANCE_ID PLOG_DEFAULT_INSTANCE
#endif

#define sLOG(severity, delta)     LOG_(SLOG_DEFAULT_INSTANCE_ID, SLOG::Delta(severity,delta))

