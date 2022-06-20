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
SBacktrace
===========

Utilities for dumping process backtraces.

Using the addresses in the debugger::


    (lldb) source list -a 0x0000000101f1be26
    /usr/local/opticks_externals/g4_1042/lib/libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() + 1334 at /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/tracking/src/G4SteppingManager2.cc:251
       246 	                                    &fGPILSelection );
       247 	#ifdef G4VERBOSE
       248 	                         // !!!!! Verbose
       249 	     if(verboseLevel>0) fVerbose->DPSLAlongStep();
       250 	#endif
    -> 251 	     if(physIntLength < PhysicalStep){
       252 	       PhysicalStep = physIntLength;
       253 	
       254 	       // Check if the process wants to be the GPIL winner. For example,
       255 	       // multi-scattering proposes Step limit, but won't be the winner.
       256 	       if(fGPILSelection==CandidateForSelection){
    (lldb) 




**/


#include "SYSRAP_API_EXPORT.hh"
#include <ostream>

struct SYSRAP_API SBacktrace
{
    static void Dump(); 
    static void DumpCaller(); 

    static void Dump(std::ostream& out) ;
    static void DumpCaller(std::ostream& out) ;

    static const char* CallSite(const char* call="::flat()" , bool addr=true );  
};



