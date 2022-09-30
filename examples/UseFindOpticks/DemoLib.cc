#include "PLOG.hh"
#include "DemoLib.hh"

const plog::Severity DemoLib::LEVEL = PLOG::EnvLevel("DemoLib", "DEBUG" ); 

void DemoLib::Dump() 
{
    LOG(error) << "[ before LOG(LEVEL) " ; 
    LOG(LEVEL) << "DemoLib::Dump" ; 
    LOG(error) << "] after LOG(LEVEL) " ; 

    std::cout << "DemoLib::Dump" << std::endl ; 
}


