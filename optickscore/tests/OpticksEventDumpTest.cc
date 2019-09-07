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

/**
OpticksEventDumpTest
======================


**/

#include "OPTICKS_LOG.hh"

#include "BOpticksKey.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    // BOpticksKey::SetKey(NULL);  // <-- makes sensitive to OPTICKS_KEY envvar 
    // this is done internally at Opticks instanciation when have argument --envkey 

    Opticks ok(argc, argv);
    ok.configure();

    bool g4 = ok.hasOpt("vizg4|evtg4") ;
    OpticksEvent* evt = ok.loadEvent(!g4);

    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }

    OpticksEventDump dump(evt);

    dump.dump(0);

    return 0 ; 
}
