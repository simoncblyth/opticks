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

// op --tevtload
// tlens-load 

#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    //NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    bool ok_ = true ; 
    unsigned tagoffset = 0 ; 

    OpticksEvent* evt = ok.loadEvent(ok_, tagoffset);

    if(!evt)
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }


    OpticksEventDump dmp(evt);
    dmp.dump(0);



    return 0 ; 
}
