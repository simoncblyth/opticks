#include <cstdlib>
#include <iostream>
#include <plog/Log.h> 

int main()
{
    const char* path = getenv("PLOGPATH") ; 
    std::cout << "writing to PLOGPATH " << path << std::endl ; 

    plog::init(plog::debug, path ); 

    LOGD << "LOGD";
    LOG_DEBUG << "LOG_DEBUG"; 
    LOG(plog::debug) << "LOG(plog::debug)" ; 

    return 0;
}


