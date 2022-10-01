#include <iostream>
#include "DEMO.hh"
#include "plog/Log.h"

void DEMO::Dump()
{
    std::cout << "[ DEMO::Dump " << std::endl ; 

    LOG(plog::none)    << "demo LOG(plog::none)    " << plog::none  ; 
    LOG(plog::fatal)   << "demo LOG(plog::fatal)   " << plog::fatal ; 
    LOG(plog::error)   << "demo LOG(plog::error)   " << plog::error ; 
    LOG(plog::warning) << "demo LOG(plog::warning) " << plog::warning ; 
    LOG(plog::info)    << "demo LOG(plog::info)    " << plog::info ; 
    LOG(plog::debug)   << "demo LOG(plog::debug)   " << plog::debug ; 
    LOG(plog::verbose) << "demo LOG(plog::verbose) " << plog::verbose ; 

    std::cout << "] DEMO::Dump " << std::endl ; 
}

