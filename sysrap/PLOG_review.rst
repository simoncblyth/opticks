PLOG review
==============

Overview
----------

PLOG provides a thin convenience/configuration layer above the 
underlying plog logging framework. In particular it 
makes it simple to setup a chained multi-logger system with a
main logger and loggers for each library with individually
controlled logging levels via command line arguments, 
which all log to the same console and logfile. 

::

    OKTest --warn \              # global loglevel
           --npy verbose \       # per-library loglevel 
           --sysrap warning 


Getting rid of the dangerous re-define of trace
-------------------------------------------------

Observe a problem with xercesc headers : so bite the bullet 
and get rid of the dangerous define and switch all 
use of LOG(trace) into LOG(verbose)

::

    epsilon:sysrap blyth$ hg diff PLOG.hh
    diff -r 3e7743e5bc48 sysrap/PLOG.hh
    --- a/sysrap/PLOG.hh    Thu Aug 16 00:01:19 2018 +0800
    +++ b/sysrap/PLOG.hh    Thu Aug 16 12:57:29 2018 +0800
    @@ -15,7 +15,7 @@
     using plog::verbose ;
     
     // hmm dangerous but what alternative 
    -#define trace plog::verbose 
    +//#define trace plog::verbose 
     
     #include "SYSRAP_API_EXPORT.hh"
     
    epsilon:sysrap blyth$ 



plog
------

* https://github.com/SergiusTheBest/plog

chained loggers
~~~~~~~~~~~~~~~~~~

* https://github.com/SergiusTheBest/plog#chained-loggers
* https://github.com/SergiusTheBest/plog/blob/master/samples/Chained/ChainedApp/Main.cpp
* https://github.com/SergiusTheBest/plog/blob/master/samples/Chained/ChainedLib/Main.cpp

A Logger can work as an Appender for another Logger. So you can chain several
loggers together. This is useful for streaming log messages from a shared
library to the main application binary.


::

    // main app

    // Functions imported form the shared library.
    extern "C" void initialize(plog::Severity severity, plog::IAppender* appender);
    extern "C" void foo();

    int main()
    {
        plog::init(plog::debug, "ChainedApp.txt"); // Initialize the main logger.

        LOGD << "Hello from app!"; // Write a log message.

        initialize(plog::debug, plog::get()); // Initialize the logger in the shared library. Note that it has its own severity.
        foo(); // Call a function from the shared library that produces a log message.

        return 0;
    }

::

    // shared library

    // Function that initializes the logger in the shared library. 
    extern "C" void EXPORT initialize(plog::Severity severity, plog::IAppender* appender)
    {
        plog::init(severity, appender); // Initialize the shared library logger.
    }

    // Function that produces a log message.
    extern "C" void EXPORT foo()
    {
        LOGI << "Hello from shared lib!";
    }


Typical Opticks executable main
----------------------------------
::


     01 // op --opticks 
      2 
      3 #include <iostream>
      4 
      5 #include "BFile.hh"
      6 #include "PLOG.hh"
      7 
      8 #include "SYSRAP_LOG.hh"
      9 #include "BRAP_LOG.hh"
     10 #include "NPY_LOG.hh"
     11 #include "OKCORE_LOG.hh"
     12 
     13 #include "Opticks.hh"
     14 
     ..
     74 int main(int argc, char** argv)
     75 {
     76     PLOG_(argc,argv);
     77     LOG(info) << argv[0] ;
     78 
     79 
     80     SYSRAP_LOG__ ;
     81     BRAP_LOG__ ;
     82     NPY_LOG__ ;
     83     OKCORE_LOG__ ;
     84 
     85 
     86     Opticks ok(argc, argv);
     87 


PLOG.hh
~~~~~~~~~~

::

    005 #include <cstddef>
      6 #include <plog/Log.h>
      7 
      8 // translate from boost log levels to plog 
      9 using plog::fatal ;
     10 using plog::error ;
     11 using plog::warning ;
     12 using plog::info ;
     13 using plog::debug ;
     14 using plog::verbose ;
     15 
     16 // hmm dangerous but what alternative 
     17 #define trace plog::verbose 
     18 
     19 #include "SYSRAP_API_EXPORT.hh"
    ...
    105 struct PLOG ;
    106 
    107 struct SYSRAP_API PLOG
    108 {
    109     int    argc ;
    110     char** argv ;
    111     int   level ;
    112     const char* logpath ;
    113     int   logmax ;
    114 
    115     PLOG(int argc, char** argv, const char* fallback="VERBOSE", const char* prefix=NULL );
    116 
    117     const char* name();
    118     int parse( const char* fallback);
    119     int parse( plog::Severity _fallback);
    120     int prefixlevel_parse( const char* fallback, const char* prefix);
    121     int prefixlevel_parse( plog::Severity _fallback, const char* prefix);
    122 
    123     static int  _parse(int argc, char** argv, const char* fallback);
    124     static int  _prefixlevel_parse(int argc, char** argv, const char* fallback, const char* prefix);
    125     static void _dump(const char* msg, int argc, char** argv);
    126     static const char* _name(plog::Severity severity);
    127     static const char* _name(int level);
    128     static const char* _logpath_parse(int argc, char** argv);
    129 
    130     static PLOG* instance ;
    ...     ^^^^^^^^^^^^^^^^^^^^^^^^^^ static singleton instance : possibly source of stomp problems with recent gcc ?

    131 };
    132 
    133 
    134 #include "PLOG_INIT.hh"
    135 


PLOG_INIT.hh
~~~~~~~~~~~~~~

::

     01 #include <plog/Log.h>
      2 #include <plog/Appenders/ColorConsoleAppender.h>
      3 #include <plog/Appenders/ConsoleAppender.h>
      4 #include <plog/Formatters/FuncMessageFormatter.h>
      5 
      6 #include "PlainFormatter.hh"
      7 
      8 /*
      9 
     10 PLOG_INIT macros are used in two situations:
     11 
     12 * an executable main as a result of PLOG_ or PLOT_COLOR applied
     13   to the arguments
     14 
     15 * package logger 
     16 
     17 
     18 */
     19 
     20 
     21 
     22 #define PLOG_INIT(level, app1, app2 ) \
     23 { \
     24     plog::IAppender* appender1 = app1 ? static_cast<plog::IAppender*>(app1) : NULL ; \
     25     plog::IAppender* appender2 = app2 ? static_cast<plog::IAppender*>(app2) : NULL ; \
     26     plog::Severity severity = static_cast<plog::Severity>(level) ; \
     27     plog::init( severity ,  appender1 ); \
     28     if(appender2) \
     29         plog::get()->addAppender(appender2) ; \
     30 } \ 
     31     
     32     
     33 #define PLOG_COLOR(argc, argv) \
     34 { \ 
     35     PLOG _plog(argc, argv); \
     36     static plog::RollingFileAppender<plog::FuncMessageFormatter> fileAppender( _plog.logpath, _plog.logmax); \
     37     static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
     38     PLOG_INIT( _plog.level, &consoleAppender, &fileAppender ); \
     39 } \ 
     40     
     41 #define PLOG_(argc, argv) \
     42 { \ 
     43     PLOG _plog(argc, argv); \
     44     static plog::RollingFileAppender<plog::FuncMessageFormatter> fileAppender( _plog.logpath, _plog.logmax); \
     45     static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender; \
     46     PLOG_INIT( _plog.level,  &consoleAppender, &fileAppender ); \
     47 } \ 
     48     
     ...


What PLOG_(argc, argv) does
------------------------------

::

     74 int main(int argc, char** argv)
     75 {
     76     PLOG_(argc,argv);
     77     LOG(info) << argv[0] ;
     78  

Instanciates PLOG struct into main which parses command line arguments into:

* global logmax
* global logpath
* holds onto arguments within PLOG::instance, for use from the package loggers

Invokes PLOG_INIT macro which instanciates the main plog logger and hooks up 
file and console appenders.


What SYSRAP_LOG__ and other pkg macros do
---------------------------------------------

::

      5 #define SYSRAP_LOG__  {     SYSRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "SYSRAP"), plog::get(), NULL );  } 

      ## notice that the main logger plog::get() is being passed to the lib logger as an appender


* uses the command line arguments persisted in PLOG::instance to define the per-package logging level 
  and passes this level to the package libs where PLOG_INIT is invoked to instanciate the 
  per-library loggers.  

* also chains together the main and library loggers ; this means than the lib 
  logger acts as an appender for the main logger


Package Loggers
-------------------

SYSRAP_LOG.hh  SYSRAP_LOG.cc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     01 
      2 #pragma once
      3 #include "SYSRAP_API_EXPORT.hh"
      4 
      5 #define SYSRAP_LOG__  {     SYSRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "SYSRAP"), plog::get(), NULL );  } 
      6 
      7 #define SYSRAP_LOG_ {     SYSRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
      8 class SYSRAP_API SYSRAP_LOG {
      9    public:
     10        static void Initialize(int level, void* app1, void* app2 );
     11        static void Check(const char* msg);
     12 };
     13 

     01 
      2 #include <plog/Log.h>
      3 
      4 #include "SYSRAP_LOG.hh"
      5 #include "PLOG_INIT.hh"
      6 #include "PLOG.hh"
      7        
      8 void SYSRAP_LOG::Initialize(int level, void* app1, void* app2 )
      9 {  
     10     PLOG_INIT(level, app1, app2); 
     11 }      
     12 void SYSRAP_LOG::Check(const char* msg)
     13 {
     14     PLOG_CHECK(msg);
     15 }
     16 


NPY_LOG.hh NPY_LOG.cc
~~~~~~~~~~~~~~~~~~~~~~~~~~

::


     01 
      2 #pragma once
      3 #include "NPY_API_EXPORT.hh"
      4 
      5 #define NPY_LOG__  {     NPY_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "NPY"), plog::get(), NULL );  } 
      6 
      7 #define NPY_LOG_ {     NPY_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
      8 class NPY_API NPY_LOG {
      9    public:
     10        static void Initialize(int level, void* app1, void* app2 );
     11        static void Check(const char* msg);
     12 };
     13 

     01 
      2 #include <plog/Log.h>
      3 
      4 #include "NPY_LOG.hh"
      5 #include "PLOG_INIT.hh"
      6 #include "PLOG.hh"
      7 
      8 void NPY_LOG::Initialize(int level, void* app1, void* app2 )
      9 {
     10     PLOG_INIT(level, app1, app2);
     11 }
     12 void NPY_LOG::Check(const char* msg)
     13 {
     14     PLOG_CHECK(msg);
     15 }





Same pattern followed by all package loggers ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    simon:opticks blyth$ find . -name '*_LOG.cc'
    ./assimprap/ASIRAP_LOG.cc
    ./boostrap/BRAP_LOG.cc
    ./cfg4/CFG4_LOG.cc
    ./cudarap/CUDARAP_LOG.cc
    ./ggeo/GGEO_LOG.cc
    ./oglrap/OGLRAP_LOG.cc
    ./ok/OK_LOG.cc
    ./okg4/OKG4_LOG.cc
    ./okop/OKOP_LOG.cc
    ./openmeshrap/MESHRAP_LOG.cc
    ./optickscore/OKCORE_LOG.cc
    ./opticksgeo/OKGEO_LOG.cc
    ./opticksgl/OKGL_LOG.cc
    ./opticksnpy/NPY_LOG.cc
    ./optixrap/OXRAP_LOG.cc
    ./sysrap/SYSRAP_LOG.cc
    ./thrustrap/THRAP_LOG.cc
    simon:opticks blyth$ 

Using the preprocessor macros allows the same logging setup code to 
be planted in every library.



