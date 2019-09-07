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
OpticksEventCompareTest
========================

See eg tboolean-torus-a


**/

#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"

#include "NCSGList.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventCompare.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    OpticksEvent* evt = ok.loadEvent();

    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }

    OpticksEvent* g4evt = ok.loadEvent(false);

    if(!g4evt || g4evt->isNoLoad())
    {
        LOG(fatal) << "failed to load g4evt " ; 
        return 0 ; 
    }

    OpticksEventCompare cf(evt,g4evt);
    cf.dump("cf(evt,g4evt)");

    return 0 ; 
}
