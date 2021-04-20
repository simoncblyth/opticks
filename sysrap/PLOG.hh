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

// NB DANGEROUS DEFINES : INCLUDE AS THE LAST HEADER
//    OTHERWISE RANDOMLY REPLACES STRINGS IN SYSTEM HEADERS

#include <cstddef>
#include <plog/Log.h>

// translate from boost log levels to plog 
using plog::fatal ;
using plog::error ;
using plog::warning ;
using plog::info ;
using plog::debug ;
using plog::verbose ;

#include "SYSRAP_API_EXPORT.hh"

/**

PLOG : Logging Infrastructure 
--------------------------------

new approach to initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks executables follow the following pattern.

.. code-block:: c


   #include "OPTICKS_LOG.hh"  // brings in separate headers for each projects logger depending on defines such as OPTICKS_GGEO

   int main(int argc, char** argv)
   {
       OPTICKS_LOG(argc, argv);  // pass arguments to PLOG_ macro

       Opticks ok(argc, argv);

       //  ...  exercise Opticks  ... 
       return 0 ; 
   }



OPTICKS_LOG is a macro from OPTICKS_LOG.hh that must be placed in the main program::

    #define OPTICKS_LOG(argc, argv) {      PLOG_COLOR(argc, argv);     OPTICKS_LOG_::Initialize(PLOG::instance, plog::get(), NULL ); } 

The PLOG_COLOR macro from PLOG_INIT.hh creates two static appenders in the main compilation unit::

    #define PLOG_COLOR(argc, argv) \
    { \
        PLOG* _plog = new PLOG(argc, argv); \
        static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
        static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
        PLOG_INIT( _plog->level, &consoleAppender, &fileAppender ); \
    } \

The PLOG_INIT macro also from PLOG_INIT.hh applies plog::init to the main consoleAppender and adds the fileAppender:: 

    #define PLOG_INIT(level, app1, app2 ) \
    { \
        plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
        plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
        plog::Severity severity = static_cast<plog::Severity>(level) ; \
        plog::init( severity ,  appender1 ); \
        if(appender2) \
            plog::get()->addAppender(appender2) ; \
    } \


The subsequent OPTICKS_LOG_::Initialize passes the primary instance returned by plog::get() in the 
main compilation unit to the static logging setup in each of the shared object libraries selected
by compilation defines such as OPTICKS_SYSRAP, OPTICKS_BRAP. 

Extracts from OPTICKS_LOG.hh showing the OPTICKS_LOG_::Initialize::

    ...
    #ifdef OPTICKS_SYSRAP
    #include "SYSRAP_LOG.hh"
    #endif
    #ifdef OPTICKS_BRAP
    #include "BRAP_LOG.hh"
    #endif
    #ifdef OPTICKS_NPY
    #include "NPY_LOG.hh"
    #endif

    ...
 
    class SYSRAP_API OPTICKS_LOG_ {
       public:
           // initialize all linked loggers and hookup the main logger
           static void Initialize(PLOG* instance, void* app1, void* app2 )
           {
               int max_level = instance->parse("info") ;
               // note : can decrease verbosity from the max_level in the subproj, but not increase
    
    #ifdef OPTICKS_SYSRAP
        SYSRAP_LOG::Initialize(instance->prefixlevel_parse( max_level, "SYSRAP"), app1, NULL );
    #endif
    #ifdef OPTICKS_BRAP
        BRAP_LOG::Initialize(instance->prefixlevel_parse( max_level, "BRAP"), app1, NULL );
    #endif
    #ifdef OPTICKS_NPY
        NPY_LOG::Initialize(instance->prefixlevel_parse( max_level, "NPY"), app1, NULL );
    #endif


From SYSRAP_LOG.cc::

    void SYSRAP_LOG::Initialize(int level, void* app1, void* app2 )
    {
        PLOG_INIT(level, app1, app2);
    }


The somewhat bizarre usage implementation based on preprocessor
macros allows the static logger symbols to be planted within 
each of the shared objects in a manner that works on Mac, Linux 
and Windows.  

The structure was based on the Chained example from my fork of the upstream plog

* https://github.com/simoncblyth/plog/tree/master/samples/Chained

The structure relies on the static plog instances in all of the libraries and 
those in the main being distinct which means that it needs 
compilation options::

    -fvisibility=hidden
    -fvisibility-inlines-hidden

These are configured in cmake/Modules/OpticksCXXFlags.cmake
The big advantage of this is that the logging level can be 
individually controlled for each of the libraries.

*PLOG* parses command line arguments and configures the 
logging level of each project, for example:

.. code-block:: sh

   OpticksResourceTest --sysrap trace --npy info   # lower cased tags identify the projects
   GGeoViewTest --npy debug    

Available logging levels are:

* *trace* :  most verbose level, providing a great deal of output  
* *debug*
* *info* : normal default logging level 
* *warning*
* *error*
* *fatal*  


The tags for each project are listed below.

=======================  ============================
 Project Folder           Tag 
=======================  ============================
              sysrap                         SYSRAP 
            boostrap                           BRAP 
          opticksnpy                            NPY 
         optickscore                         OKCORE 
                ggeo                           GGEO 
           assimprap                         ASIRAP 
         openmeshrap                        MESHRAP 
          opticksgeo                          OKGEO 
              oglrap                         OGLRAP 
             cudarap                        CUDARAP 
           thrustrap                          THRAP 
            optixrap                          OXRAP 
           opticksop                           OKOP 
           opticksgl                           OKGL 
                  ok                             OK 
                cfg4                           CFG4 
=======================  ============================







older low level approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This old approach has been replaced by the OPTICKS_LOG 
approach described above, although some test executables
have yet to be updated. 

.. code-block:: c


   #include "SYSRAP_LOG.hh"  // headers for each projects logger
   #include "BRAP_LOG.hh"
   #include "NPY_LOG.hh"
   #include "OKCORE_LOG.hh"

   #include "PLOG.hh"        // infrastructure header

   int main(int argc, char** argv)
   {
       PLOG_(argc, argv);  // pass arguments to PLOG_ macro

       SYSRAP_LOG__ ;     // setup loggers for all projects you want to see output from
       BRAP_LOG__ ; 
       NPY_LOG__ ;       
       OKCORE_LOG__ ;       

       Opticks ok(argc, argv);

       //  ...  exercise Opticks  ... 

       return 0 ; 
   }



**/

struct PLOG ; 



/**
Delta
------
::

    enum Severity
    {   
        none = 0,
        fatal = 1,
        error = 2,
        warning = 3,
        info = 4,
        debug = 5,
        verbose = 6 
    };  

**/




#include "SAr.hh"
struct STTF ; 

struct SYSRAP_API PLOG 
{
    static const int MAXARGC ; 

    SAr         args ; 
    STTF*       ttf ;    // truetypefont

    int         level ; 
    const char* filename ; 
    int         maxFileSize ;    // bytes
    int         maxFiles ; 

    static plog::Severity Delta(plog::Severity level_, int delta); 
    static plog::Severity EnvLevel( const char* key, const char* fallback); 

    PLOG(const char* name, const char* fallback="VERBOSE", const char* prefix=NULL );
    PLOG(int argc, char** argv, const char* fallback="VERBOSE", const char* prefix=NULL );
    void init(const char* fallback, const char* prefix); 

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


    static PLOG* instance ; 
};

#define pLOG(severity, delta)     LOG_(PLOG_DEFAULT_INSTANCE, PLOG::Delta(severity,delta))

#include "PLOG_INIT.hh"


