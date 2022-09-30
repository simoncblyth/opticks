#include "PLOG.hh"
#include "DemoLib.hh"

const plog::Severity DemoLib::LEVEL = PLOG::EnvLevel("DemoLib", "DEBUG" ); 

void DemoLib::Dump() 
{
    LOG(error) << "[ before LOG(LEVEL) " ; 
    LOG(LEVEL) << "DemoLib::Dump" ; 
    LOG(error) << "] after LOG(LEVEL) " ; 
    std::cout << "DemoLib::Dump" << std::endl ; 

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


    std::cout << "LOG(LEVEL) " << std::endl ; 
    LOG(LEVEL)    << " LOG(LEVEL)   " << LEVEL   ; 

    plog::Logger<0>* instance = plog::get<0>() ; 
    plog::Severity logmax = instance->getMaxSeverity() ; 

    std::cout 
         << " plog::Logger<0>* instance " << instance << std::endl 
         << " logmax " << logmax << " plog::severityToString(logmax) " << plog::severityToString(logmax) << std::endl
         << " LEVEL  " << LEVEL  << " plog::severityToString(LEVEL)  " << plog::severityToString(LEVEL)  << std::endl
         ; 

}


