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

#include "CFG4_BODY.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksEvent.hh"
#include "OpticksHub.hh"
#include "OpticksRun.hh"
#include "OpticksGen.hh"

#include "CG4.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    LOG(warning) << " post hub " ; 

    OpticksRun* run = ok.getRun();
    LOG(warning) << " post run " ; 
    OpticksGen* gen = hub.getGen();
    
    CG4* g4 = new CG4(&hub) ; 
    LOG(warning) << " post CG4 " ; 

    g4->interactive();

    LOG(warning) << "  post CG4::interactive"  ;


    run->createEvent();

    if(ok.isFabricatedGensteps())  // eg TORCH running
    { 
        NPY<float>* gs = gen->getInputGensteps() ;
        LOG(error) << " setting gensteps " << gs ; 
        run->setGensteps(gs);
    }
    else
    {
        LOG(error) << " not setting gensteps " ; 
    }

    g4->propagate();

    LOG(info) << "  CG4 propagate DONE "  ;

    ok.postpropagate();

    OpticksEvent* evt = run->getG4Event();
    assert(evt->isG4()); 
    evt->save();

    LOG(info) << "  evt save DONE "  ;

    g4->cleanup();




    LOG(info) << "exiting " << argv[0] ; 

    return 0 ; 
}

