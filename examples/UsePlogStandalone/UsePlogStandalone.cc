#include <cstdlib>
#include <iostream>

#include <plog/Log.h> 
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>

#include "DEMO.hh"
#include "DEMO_LOG.hh"

int main()
{
    const char* path = getenv("PLOGPATH") ? getenv("PLOGPATH") : "UsePlogStandalone.log" ; 
    int maxFileSize = 10000 ; 
    int maxFiles = 1 ; 

    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( path, maxFileSize, maxFiles );
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; 

    //plog::Severity loglev = plog::verbose ; 
    //plog::Severity loglev = plog::info ; 
    plog::Severity loglev = plog::error ; 

    std::cout << " loglev " << loglev << " plog::severityToString(loglev) " << plog::severityToString(loglev) << std::endl ; 

    plog::Logger<0>& log = plog::init(loglev); 
    log.addAppender( &consoleAppender ); 
    log.addAppender( &fileAppender ); 

    DEMO_LOG::Initialize( loglev, &log );  


    LOG(plog::none)    << "main LOG(plog::none)    " << plog::none  ; 
    LOG(plog::fatal)   << "main LOG(plog::fatal)   " << plog::fatal ; 
    LOG(plog::error)   << "main LOG(plog::error)   " << plog::error ; 
    LOG(plog::warning) << "main LOG(plog::warning) " << plog::warning ; 
    LOG(plog::info)    << "main LOG(plog::info)    " << plog::info ; 
    LOG(plog::debug)   << "main LOG(plog::debug)   " << plog::debug ; 
    LOG(plog::verbose) << "main LOG(plog::verbose) " << plog::verbose ; 


    DEMO::Dump(); 


    return 0;
}


