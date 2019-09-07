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

#include "Opticks.hh"
#include "OpticksEvent.hh"

#include "TrivialCheckNPY.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    int multi = ok.getMultiEvent();

    LOG(info) << argv[0] 
              << " multi " << multi 
              ; 

    int fail(0);

    for(int tagoffset=0 ; tagoffset < multi ; tagoffset++)
    {
        LOG(fatal) << " ################## tagoffset " << tagoffset ; 

        OpticksEvent* evt = ok.loadEvent(true, tagoffset);  
        LOG(info) << " dir " << evt->getDir() ; 


        if(evt->isNoLoad()) 
        {
            LOG(error) << "FAILED to load evt from " << evt->getDir() ;
            continue ;  
        }
         
        evt->Summary();

        char entryCode = evt->getEntryCode() ; 
        if(!TrivialCheckNPY::IsApplicable( entryCode )) 
        {
            LOG(error) << " skipping event with non-applicable entryCode " << entryCode ; 
            continue ; 
        }


        TrivialCheckNPY tcn(evt->getPhotonData(), evt->getGenstepData(), entryCode );
        fail += tcn.check(argv[0]);
    }

    LOG(info) << " fails: " << fail ; 
    assert(fail == 0);
 
    return 0 ; 
}

/**

TrivialTest
=============

Checks correspondence between input gensteps and the photon
buffer output by the trivial entry point, which does
minimal processing just checking that genstep seeds are
correctly uploaded and available at photon level in the OptiX 
program.

Produce single event to examine and check it with::

   OKTest --compute --save --trivial     ## default to torch 
   TrivialTest  

   OKTest --compute --cerenkov --save --trivial
   TrivialTest --cerenkov


For multievent (1 is default anyhow so this is same as above)::

   OKTest --cerenkov --trivial --save --compute --multievent 1
   TrivialTest --cerenkov --multievent 1


See :doc:`notes/issues/geant4_opticks_integration/multi_event_seed_stuck_at_zero_for_second_event`

   

**/

