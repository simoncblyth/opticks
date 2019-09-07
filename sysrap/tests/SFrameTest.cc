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

// TEST=SFrameTest om-t 

#include <string>
#include "OPTICKS_LOG.hh"
#include "SFrame.hh"

std::string macOS = R"(
4   libG4processes.dylib                0x00000001090baee5 _ZN12G4VEmProcess36PostStepGetPhysicalInteractionLengthERK7G4TrackdP16G4ForceCondition + 661
5   libG4tracking.dylib                 0x00000001088ffff0 _ZN10G4VProcess12PostStepGPILERK7G4TrackdP16G4ForceCondition + 80
6   libG4tracking.dylib                 0x00000001088ffa1a _ZN17G4SteppingManager24DefinePhysicalStepLengthEv + 298
7   libG4tracking.dylib                 0x00000001088fcc3a _ZN17G4SteppingManager8SteppingEv + 394
8   libG4tracking.dylib                 0x000000010891386f _ZN17G4TrackingManager15ProcessOneTrackEP7G4Track + 1679
9   libG4event.dylib                    0x00000001087da71a _ZN14G4EventManager12DoProcessingEP7G4Event + 3306
10  libG4event.dylib                    0x00000001087dbc2f _ZN14G4EventManager15ProcessOneEventEP7G4Event + 47
11  libG4run.dylib                      0x00000001086e79f5 _ZN12G4RunManager15ProcessOneEventEi + 69
12  libG4run.dylib                      0x00000001086e7825 _ZN12G4RunManager11DoEventLoopEiPKci + 101
13  libG4run.dylib                      0x00000001086e5ce1 _ZN12G4RunManager6BeamOnEiPKci + 193
14  libCFG4.dylib                       0x0000000106a63df9 _ZN3CG49propagateEv + 1689
15  libOKG4.dylib                       0x00000001000e22b6 _ZN7OKG4Mgr10propagate_Ev + 182
16  libOKG4.dylib                       0x00000001000e1ec6 _ZN7OKG4Mgr9propagateEv + 470
17  OKG4Test                            0x0000000100014c89 main + 489
18  libdyld.dylib                       0x00007fff6bd8b015 start + 1
19  ???                                 0x0000000000000005 0x0 + 5
)" ; 

std::string Linux = R"(
/home/blyth/local/opticks/lib64/libSysRap.so(+0x10495) [0x7fffe54cf495] 
/home/blyth/local/opticks/lib64/libSysRap.so(_ZN10SBacktrace4DumpEv+0x1b) [0x7fffe54cf823] 
/home/blyth/local/opticks/lib64/libCFG4.so(_ZN10CMixMaxRng4flatEv+0x138) [0x7ffff7955f84] 
/home/blyth/local/opticks/externals/lib64/libG4processes.so(_ZN12G4VEmProcess36PostStepGetPhysicalInteractionLengthERK7G4TrackdP16G4ForceCondition+0x2ce) [0x7ffff1e3f21a] 
/home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN10G4VProcess12PostStepGPILERK7G4TrackdP16G4ForceCondition+0x42) [0x7ffff36ff9b2] 
/home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN17G4SteppingManager24DefinePhysicalStepLengthEv+0x127) [0x7ffff36fe161] 
/home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN17G4SteppingManager8SteppingEv+0x1c2) [0x7ffff36fb410] 
/home/blyth/local/opticks/externals/lib64/libG4tracking.so(_ZN17G4TrackingManager15ProcessOneTrackEP7G4Track+0x284) [0x7ffff3707236] 
/home/blyth/local/opticks/externals/lib64/libG4event.so(_ZN14G4EventManager12DoProcessingEP7G4Event+0x4ce) [0x7ffff397fd46] 
/home/blyth/local/opticks/externals/lib64/libG4event.so(_ZN14G4EventManager15ProcessOneEventEP7G4Event+0x2e) [0x7ffff3980572] 
/home/blyth/local/opticks/externals/lib64/libG4run.so(_ZN12G4RunManager15ProcessOneEventEi+0x57) [0x7ffff3c82665] 
/home/blyth/local/opticks/externals/lib64/libG4run.so(_ZN12G4RunManager11DoEventLoopEiPKci+0x59) [0x7ffff3c824d7] 
/home/blyth/local/opticks/externals/lib64/libG4run.so(_ZN12G4RunManager6BeamOnEiPKci+0xc1) [0x7ffff3c81d2d] 
/home/blyth/local/opticks/lib/CerenkovMinimal() [0x41a014] 
/home/blyth/local/opticks/lib/CerenkovMinimal() [0x419ed1] 
/home/blyth/local/opticks/lib/CerenkovMinimal() [0x4098bd] 
/usr/lib64/libc.so.6(__libc_start_main+0xf5) [0x7fffe00e1445] 
/home/blyth/local/opticks/lib/CerenkovMinimal() [0x409629] 
)" ; 


int main(int  argc, char** argv )
{
    OPTICKS_LOG(argc, argv); 

#ifdef __APPLE__
    const char* lines = macOS.c_str(); 
#else
    const char* lines = Linux.c_str(); 
#endif
    LOG(info) << std::endl << lines  ; 


    std::istringstream iss(lines);
    std::string line ;
    while (getline(iss, line, '\n'))
    {   
        if(line.empty()) continue ; 

        //std::cout << "[" << line << "]" << std::endl ; 

        SFrame f((char*)line.c_str());
        f.dump(); 
    }   

    return 0 ; 
}

