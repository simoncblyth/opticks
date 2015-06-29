#include "ThrustEngine.hh"

#include "stdio.h"
#include <thrust/version.h>
#include <iostream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void vers()
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    printf(" major %d minor %d \n", major, minor);

    LOG(info) << "Thrust v" << major << "." << minor << std::endl;
}


void ThrustEngine::version()
{
   vers();
}







