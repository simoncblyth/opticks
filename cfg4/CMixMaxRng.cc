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

#include <fstream>

#include "SPath.hh"
#include "Randomize.hh"
#include "CMixMaxRng.hh"
#include "SBacktrace.hh"
#include "PLOG.hh"


CMixMaxRng::CMixMaxRng()
    :
    count(0), 
    out(NULL)
{
    CLHEP::HepRandom::setTheEngine( this );  

    //out = new std::ostream(std::cout.rdbuf()) ;

    const char* path = SPath::Resolve("$TMP/simstream.txt", 1);  // 1:assume filepath argument
    out = new std::ofstream(path) ;

}


/**
CMixMaxRng::flat
-------------------

Instrumented shim for flat,  

Finding the CallSite in the backtrace with "::flat" 
matches either ::flat() or ::flatArray(.. 
so get the line following those.


**/

double CMixMaxRng::flat()
{
    double v = CLHEP::MixMaxRng::flat(); 

    if(count == 0)
        SBacktrace::Dump();

    const char* caller = SBacktrace::CallSite( "::flat" ) ; 

    (*out) 
        << std::setw(6) << count 
        << " : " 
        << std::setw(10) << std::fixed << v 
        << " : "
        << caller
        << std::endl 
        ;

    count += 1 ; 
    return v ;   
}


void CMixMaxRng::preTrack() 
{
    LOG(info) << "." ; 
} 
void CMixMaxRng::postTrack()
{
    LOG(info) << "." ; 
} 
void CMixMaxRng::postStep() 
{
    LOG(info) << "." ; 
}
void CMixMaxRng::postpropagate()
{
    LOG(info) << "." ; 
}
double CMixMaxRng::flat_instrumented(const char* file, int line)
{
    LOG(info) << file << ":" << line ; 
    return flat(); 
}



