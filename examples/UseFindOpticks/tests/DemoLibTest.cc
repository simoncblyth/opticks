/**

As LOCAL_OPTICKS_LOG is used in the main it gets the main appended with plog::get() 
and passes that to the shared libs 



160 class SYSRAP_API OPTICKS_LOG_ {
161    public:
162        // initialize all linked loggers and hookup the main logger
163        static void Initialize(PLOG* instance, void* app1, void* )
164        {
165            int max_level = instance->parse("info") ;
166            // note : can decrease verbosity from the max_level in the subproj, but not increase
167 
168 #ifdef OPTICKS_SYSRAP
169     SYSRAP_LOG::Initialize(instance->prefixlevel_parse( max_level, "SYSRAP"), app1, NULL );
170 #endif


Note that this is organized such that the PLOG_INIT happens within the shared lib, not the main::

 27 class SYSRAP_API SYSRAP_LOG {
 28    public:
 29        static void Initialize(int level, void* app1, void* app2 );
 30        static void Check(const char* msg);
 31 };

 27 void SYSRAP_LOG::Initialize(int level, void* app1, void* app2 )
 28 {
 29     PLOG_INIT(level, app1, app2);
 30 }

 65 #define PLOG_INIT(level, app1, app2 ) \
 66 { \
 67     plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
 68     plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
 69     plog::Severity severity = static_cast<plog::Severity>(level) ; \
 70     plog::init( severity ,  appender1 ); \
 71     if(appender2) \
 72         plog::get()->addAppender(appender2) ; \
 73 } \
 74 


Chained Loggers
------------------

* https://github.com/simoncblyth/plog/#chained-loggers

A Logger can work as an Appender for another Logger. So you can chain several
loggers together. This is useful for streaming log messages from a shared
library to the main application binary.

Shared Lib::

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


Main app::

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


**/

#include "DemoLib.hh"
#include "DEMO_LOG.hh"
#include "OPTICKS_LOG.hh"

#define LOCAL_OPTICKS_LOG(argc, argv) {  PLOG_COLOR(argc, argv); OPTICKS_LOG_::Initialize(PLOG::instance, plog::get(), NULL ); } 



int main(int argc, char** argv)
{
    LOCAL_OPTICKS_LOG(argc, argv); 

    // the logmax controls logging from the shared lib, not the main.  WHY ?  
    // inhibition of logging only works in the shared libs, not the main
    // Actually thats not much of a problem, as most everything happens in shared libs. 

    //plog::Severity logmax = plog::none  ; // 0
    //plog::Severity logmax = fatal  ;      // 1 
    //plog::Severity logmax = error  ;      // 2 
    //plog::Severity logmax = warning  ;    // 3 
    plog::Severity logmax = info  ;       // 4 
    //plog::Severity logmax = debug   ;     // 5 
    //plog::Severity logmax = verbose   ;   // 6 

    std::cout 
        << "main "
        << " logmax " << logmax 
        << " plog::severityToString(logmax) " << plog::severityToString(logmax) 
        << std::endl 
        ;

    DEMO_LOG::Initialize(logmax, plog::get(), nullptr );


    LOG(error) << "[" << argv[0] ; 
    DemoLib::Dump(); 
    LOG(error) << "]" << argv[0] ; 




    std::cout << "LOG(plog::none) " << std::endl ; 
    LOG(plog::none)     << " LOG(plog::none)    " << plog::none    ; 

    std::cout << "LOG(fatal) " << std::endl ; 
    LOG(fatal)    << " LOG(fatal)   " << fatal   ; 

    std::cout << "LOG(error) " << std::endl ; 
    LOG(error)    << " LOG(error)   " << error   ; 

    std::cout << "LOG(warning) " << std::endl ; 
    LOG(warning)  << " LOG(warning) " << warning ; 

    std::cout << "LOG(info) " << std::endl ; 
    LOG(info)     << " LOG(info)    " << info    ; 

    std::cout << "LOG(debug) " << std::endl ; 
    LOG(debug)    << " LOG(debug)   " << debug   ; 

    std::cout << "LOG(verbose) " << std::endl ; 
    LOG(verbose)  << " LOG(verbose) " << verbose ; 







    return 0 ; 

}
