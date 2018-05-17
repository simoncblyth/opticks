
// https://github.com/SergiusTheBest/plog#introduction

#include <iostream>
#include <plog/Log.h> // Step1: include the header.

int main()
{
    const char* path = "/tmp/UsePLog.log" ;
 
    std::cout << "writing to " << path << std::endl ; 

    plog::init(plog::debug, path ); // Step2: initialize the logger.

    // Step3: write log messages using a special macro. 
    // There are several log macros, use the macro you liked the most.

    LOGD << "Hello log!"; // short macro
    LOG_DEBUG << "Hello log!"; // long macro
    LOG(plog::debug) << "Hello log!"; // function-style macro

    return 0;
}
