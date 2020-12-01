/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
//#define WITH_ALIGN_DEV_DEBUG 1
//#define WITH_REFLECT_CHEAT_DEBUG 1

#define WITH_LOGDOUBLE 1
//#define WITH_LOGDOUBLE_ALT 1

#define WITH_KLUDGE_FLAT_ZERO_NOPEEK 1
//#define WITH_EXCEPTION 1 


//#define WITH_ANGULAR 1

//#define WITH_DEBUG_BUFFER 1


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
#elif WITH_LOGDOUBLE_ALT
    ss << "WITH_LOGDOUBLE_ALT " ;   
#endif


#ifdef WITH_KLUDGE_FLAT_ZERO_NOPEEK
    ss << "WITH_KLUDGE_FLAT_ZERO_NOPEEK " ;   
#endif

#ifdef WITH_EXCEPTION
    ss << "WITH_EXCEPTION " ;   
#endif

#ifdef WITH_ANGULAR
    ss << "WITH_ANGULAR " ;   
#endif

#ifdef WITH_DEBUG_BUFFER
    ss << "WITH_DEBUG_BUFFER " ;   
#endif


    std::string switches = ss.str(); 
    return switches  ; 
}

#endif


