#pragma once 

/**
OpticksSwitches.h
====================

On the one hand using switches from okc into oxrap is too much action at a distance 
BUT need to get these settings into event metadata somehow.
Best to do that near where used.

NB when searching for switches in python include the space at the end, eg:: 

    a.parameters["Switches"].find("WITH_ALIGN_DEV ") > -1  

**/


#define WITH_SEED_BUFFER 1 
#define WITH_RECORD 1 
#define WITH_SOURCE 1 
#define WITH_ALIGN_DEV 1
#define WITH_ALIGN_DEV_DEBUG 1
//#define WITH_REFLECT_CHEAT_DEBUG 1
#define WITH_LOGDOUBLE 1



#ifndef __CUDACC__

#include <sstream>
#include <string>

static std::string OpticksSwitches()
{ 
    std::stringstream ss ; 
#ifdef WITH_SEED_BUFFER
    ss << "WITH_SEED_BUFFER " ;   
#endif
#ifdef WITH_RECORD
    ss << "WITH_RECORD " ;   
#endif
#ifdef WITH_SOURCE
    ss << "WITH_SOURCE " ;   
#endif
#ifdef WITH_ALIGN_DEV
    ss << "WITH_ALIGN_DEV " ;   
#endif
#ifdef WITH_ALIGN_DEV_DEBUG
    ss << "WITH_ALIGN_DEV_DEBUG " ;   
#endif
#ifdef WITH_REFLECT_CHEAT_DEBUG
    ss << "WITH_REFLECT_CHEAT_DEBUG " ;   
#endif
#ifdef WITH_LOGDOUBLE
    ss << "WITH_LOGDOUBLE " ;   
#endif
    std::string switches = ss.str(); 
    return switches  ; 
}

#endif


