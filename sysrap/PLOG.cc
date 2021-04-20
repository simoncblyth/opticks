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

#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>


#include "PLOG.hh"
#include "SSys.hh"
#include "SProc.hh"

#define STTF_IMPLEMENTATION 1 
#include "STTF.hh"

PLOG* PLOG::instance = NULL ; 

const int PLOG::MAXARGC = 100 ;  

//#define PLOG_DBG 1

/**


plog enum levels

 fatal = 1,
 error = 2,
 warning = 3,
 info = 4,
 debug = 5,
 verbose = 6 

**/

plog::Severity PLOG::Delta(plog::Severity level_, int delta)
{
    int level = (int)level_ + delta ; 
    if(level < (int)fatal)   level = (int)fatal ; 
    if(level > (int)verbose) level = (int)verbose ; 
    return (plog::Severity)level ; 
}

/**
PLOG::EnvLevel
----------------

Used to make static logging LEVEL initializers sensitive to envvars
holding logging level strings::

    const plog::Severity ClassName::LEVEL = PLOG::EnvLevel("ClassName", "DEBUG") ;  

Static initializers run very early, prior to logging being setup.

**/

plog::Severity PLOG::EnvLevel( const char* key, const char* fallback)
{
    const char* level = SSys::getenvvar(key, fallback);  
    plog::Severity severity = plog::severityFromString(level) ;

    if(strcmp(level, fallback) != 0)
    {
        std::cout 
            << "PLOG::EnvLevel"
            << " adjusting loglevel by envvar  "
            << " key " << key  
            << " level " << level
            << " fallback " << fallback
            << std::endl 
            ;     
    }
    return severity ; 
} 




void PLOG::_dump(const char* msg, int argc, char** argv)
{
    std::cerr <<  msg
              << " argc " << argc ;

    for(int i=0 ; i < argc ; i++) std::cerr << argv[i] ; 
    std::cerr << std::endl ;               
}

int PLOG::_parse(int argc, char** argv, const char* fallback)
{
    // Parse arguments case insensitively looking for --VERBOSE --info --error etc.. returning global logging level

    assert( argc < MAXARGC && " argc sanity check fail "); 

    std::string ll = fallback ; 
    for(int i=1 ; i < argc ; ++i )
    {
        std::string arg(argv[i] ? argv[i] : "");
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        if(arg.compare("--trace")==0)   ll = "VERBOSE" ; 
        if(arg.compare("--verbose")==0) ll = "VERBOSE" ; 
        if(arg.compare("--debug")==0)   ll = "DEBUG" ; 
        if(arg.compare("--info")==0)    ll = "INFO" ; 
        if(arg.compare("--warning")==0) ll = "WARNING" ; 
        if(arg.compare("--error")==0)   ll = "ERROR" ; 
        if(arg.compare("--fatal")==0)   ll = "FATAL" ;

        // severityFromString uses first char 
    }
     
    std::transform(ll.begin(), ll.end(), ll.begin(), ::toupper);
    plog::Severity severity = plog::severityFromString(ll.c_str()) ;

    int level = static_cast<int>(severity); 

    //_dump("PLOG::parse", argc, argv );

    return level ;  
}


/**
PLOG::_logpath_parse_problematic
--------------------------------------

Constructs logfile path based on executable name argv[0] with .log appended 

This approach is problematic as when running from an absolute 
path such as with::

    gdb $(which OKTest)

This will yield a logpath within the directory where executables
are installed which presents a permission problems with shared installs.

**/

const char* PLOG::_logpath_parse_problematic(int argc, char** argv)
{
    assert( argc < MAXARGC && " argc sanity check fail "); 
    std::string lp(argc > 0 ? argv[0] : "default") ; 
    lp += ".log" ; 
    return strdup(lp.c_str());
}


/**
PLOG::_logpath()
------------------

This uses just the executable name with .log appended 

**/

const char* PLOG::_logpath()
{
    const char* exename = SProc::ExecutableName() ; 
    std::string lp(exename) ; 
    lp += ".log" ; 
    return strdup(lp.c_str());
}




/**
PLOG::_prefixlevel_parse
---------------------------

Example commandline::

   --okcore info --sysrap error --brap trace --npy trace

Parse commandline to find project logging level  
looking for a single project prefix, eg 
with the below commandline and prefix of sysrap
the level "error" should be set.

When no level is found the fallback level is used.
     
Both prefix and the arguments are lowercased before comparison.


**/

int PLOG::_prefixlevel_parse(int argc, char** argv, const char* fallback, const char* prefix)
{

    assert( argc < MAXARGC && " argc sanity check fail "); 

    std::string pfx(prefix);
    std::transform(pfx.begin(), pfx.end(), pfx.begin(), ::tolower);
    std::string apfx("--");
    apfx += pfx ;  

    std::string ll(fallback) ;
    for(int i=1 ; i < argc ; ++i )
    {
        char* ai = argv[i] ;
        char* aj = i + 1 < argc ? argv[i+1] : NULL ; 

        std::string arg(ai ? ai : "");
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        //std::cerr << arg << std::endl ; 

        if(arg.compare(apfx) == 0 && aj != NULL ) ll.assign(aj) ;
    }

    std::transform(ll.begin(), ll.end(), ll.begin(), ::toupper);

    const char* llc = ll.c_str();
    plog::Severity severity = strcmp(llc, "TRACE")==0 ? plog::severityFromString("VERB") : plog::severityFromString(llc) ;
    int level = static_cast<int>(severity); 

    //_dump("PLOG::prefix_parse", argc, argv );

    return level ; 
}




int PLOG::parse(plog::Severity _fallback)
{
    const char* fallback = _name(_fallback);
    return parse(fallback);
}
int PLOG::parse(const char* fallback)
{
    int ll = _parse(args._argc, args._argv, fallback);

#ifdef PLOG_DBG
    std::cerr << "PLOG::parse"
              << " fallback " << fallback
              << " level " << ll 
              << " name " << _name(ll)
              << std::endl ;
#endif

    return ll ; 
}


int PLOG::prefixlevel_parse(int _fallback, const char* prefix)
{
    plog::Severity fallback = static_cast<plog::Severity>(_fallback); 
    return prefixlevel_parse(fallback, prefix) ; 
}

int PLOG::prefixlevel_parse(plog::Severity _fallback, const char* prefix)
{
    const char* fallback = _name(_fallback);
    return prefixlevel_parse(fallback, prefix) ; 
}
int PLOG::prefixlevel_parse(const char* fallback, const char* prefix)
{
    int ll =  _prefixlevel_parse(args._argc, args._argv, fallback, prefix);

#ifdef PLOG_DBG
    std::cerr << "PLOG::prefixlevel_parse"
              << " fallback " << fallback
              << " prefix " << prefix 
              << " level " << ll 
              << " name " << _name(ll)
              << std::endl ;
#endif

    return ll ; 
}



const char* PLOG::_name(int level)
{
   plog::Severity severity  = static_cast<plog::Severity>(level); 
   return plog::severityToString(severity);
}
const char* PLOG::_name(plog::Severity severity)
{
   return plog::severityToString(severity);
}

const char* PLOG::name()
{
   plog::Severity severity  = static_cast<plog::Severity>(level); 
   return _name(severity);
}

const char* PLOG::exename() const 
{
    return args.exename(); 
}
const char* PLOG::cmdline() const 
{
    return args.cmdline(); 
}


const char* PLOG::get_arg_after(const char* option, const char* fallback) const 
{
    return args.get_arg_after(option, fallback); 
}
int PLOG::get_int_after(const char* option, const char* fallback) const 
{
    return args.get_int_after(option, fallback); 
}
bool PLOG::has_arg(const char* arg) const 
{
    return args.has_arg(arg); 
}






PLOG::PLOG(const char* name, const char* fallback, const char* prefix)
    :
    args(name, "OPTICKS_LOG_ARGS" , ' '),   // when argc_ is 0 the named envvar is checked for arguments instead 
    ttf(new STTF),
    level(info),
    filename(_logpath()),
    maxFileSize(SSys::getenvint("OPTICKS_LOG_MAXFILESIZE", 5000000)),
    maxFiles(SSys::getenvint("OPTICKS_LOG_MAXFILES", 10))
{
    init(fallback, prefix); 
}

PLOG::PLOG(int argc_, char** argv_, const char* fallback, const char* prefix)
    :
    args(argc_, argv_, "OPTICKS_LOG_ARGS" , ' '),   // when argc_ is 0 the named envvar is checked for arguments instead 
    ttf(new STTF),
    level(info),
    filename(_logpath()),
    maxFileSize(SSys::getenvint("OPTICKS_LOG_MAXFILESIZE", 5000000)),
    maxFiles(SSys::getenvint("OPTICKS_LOG_MAXFILES", 10))
{
    init(fallback, prefix); 
}


void PLOG::init(const char* fallback, const char* prefix)
{
    level = prefix == NULL ?  parse(fallback) : prefixlevel_parse(fallback, prefix ) ;    
    assert( instance == NULL && "ONLY EXPECTING A SINGLE PLOG INSTANCE" );
    instance = this ; 
}



