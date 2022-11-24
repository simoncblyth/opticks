SLOG : Logging Infrastructure 
=================================

This was formerly in SLOG.hh but as touching 
that header causes a full build of everything 
have moved docs separate. 

new approach to initialization
---------------------------------

Opticks executables follow the following pattern.

.. code-block:: c


   #include "OPTICKS_LOG.hh"  // brings in separate headers for each projects logger depending on defines such as OPTICKS_GGEO

   int main(int argc, char** argv)
   {
       OPTICKS_LOG(argc, argv);  // pass arguments to SLOG_ macro

       Opticks ok(argc, argv);

       //  ...  exercise Opticks  ... 
       return 0 ; 
   }



OPTICKS_LOG is a macro from OPTICKS_LOG.hh that must be placed in the main program::

    #define OPTICKS_LOG(argc, argv) {      SLOG_COLOR(argc, argv);     OPTICKS_LOG_::Initialize(SLOG::instance, plog::get(), NULL ); } 

The SLOG_COLOR macro from SLOG_INIT.hh creates two static appenders in the main compilation unit::

    #define SLOG_COLOR(argc, argv) \
    { \
        SLOG* _plog = new SLOG(argc, argv); \
        static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
        static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
        SLOG_INIT( _plog->level, &consoleAppender, &fileAppender ); \
    } \

The SLOG_INIT macro also from SLOG_INIT.hh applies plog::init to the main consoleAppender and adds the fileAppender:: 

    #define SLOG_INIT(level, app1, app2 ) \
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
           static void Initialize(SLOG* instance, void* app1, void* app2 )
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
        SLOG_INIT(level, app1, app2);
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

Notice that without this hidden visibility you will get perplexing failures, of form::

    CSGOptiXSimtraceTest: /data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include/plog/Logger.h:22: plog::Logger<instance>& plog::Logger<instance>::addAppender(plog::IAppender*) [with     int instance = 0]: Assertion `appender != this' failed.


For details see notes/issues/plog-appender-not-equal-this-assert.rst



*SLOG* parses command line arguments and configures the 
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

   #include "SLOG.hh"        // infrastructure header

   int main(int argc, char** argv)
   {
       SLOG_(argc, argv);  // pass arguments to SLOG_ macro

       SYSRAP_LOG__ ;     // setup loggers for all projects you want to see output from
       BRAP_LOG__ ; 
       NPY_LOG__ ;       
       OKCORE_LOG__ ;       

       Opticks ok(argc, argv);

       //  ...  exercise Opticks  ... 

       return 0 ; 
   }




Changing SLOG formatter 
-------------------------

Sometimes when debugging prefer shorter logging::

    epsilon:Formatters blyth$ opticks-f MessageOnlyFormatter
    ./sysrap/tests/PLogTest.cc:#include <plog/Formatters/MessageOnlyFormatter.h>
    ./sysrap/tests/PLogTest.cc:    //typedef plog::MessageOnlyFormatter FMT ; 
    ./sysrap/SLOG_INIT.hh:#include <plog/Formatters/MessageOnlyFormatter.h>
    ./sysrap/SLOG_INIT.hh://typedef plog::MessageOnlyFormatter FMT ;   // really minimal 
    epsilon:opticks blyth$ 


Can do that by changing the typedef in SLOG_INIT.hh::

     31 typedef plog::FuncMessageFormatter FMT ;     // useful to avoid dates and pids when comparing logs
     32 //typedef plog::MessageOnlyFormatter FMT ;   // really minimal 
     33 //typedef plog::TxtFormatter         FMT ;   // default full format 
     34 //typedef plog::CsvFormatter         FMT ;   // semicolon delimited full format  
     35 

Unfortunately that forces full recompile of everything 



