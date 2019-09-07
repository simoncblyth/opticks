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

#include <string>
#include <sstream>

#include "SSys.hh"    // sysrap-

#include "NPY.hpp"      // npy-
#include "FabStepNPY.hpp"

#include "Opticks.hh"   // okc-
#include "OpticksPhoton.h"   
#include "OpticksEvent.hh"

#include "OpticksHub.hh"  // okg-
#include "OpticksIdx.hh"  
#include "OpticksRun.hh"  
#include "OpticksGen.hh"  

#include "OContext.hh"   // optixrap-
#include "OPropagator.hh"   
#include "OScene.hh" 
#include "OEvent.hh" 

#include "OpSeeder.hh"   // opop-
#include "OpticksViz.hh"  // oglrap-

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"

#include "PLOG.hh"

/**
VizTest
=============

Compare with opticksop-/tests/OpSeederTest 
This adds OpticksViz handling for interop mode testing.

**/

int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 
    OXRAP_LOG__ ; 
    OKOP_LOG__ ; 

 
    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OpticksIdx idx(&hub);
    OpticksGen* gen = hub.getGen();


    OScene scene(&hub);
    OContext* octx = scene.getOContext();
    OEvent oevt(&hub, octx );
    OpSeeder  seeder(&hub, &oevt );
    OPropagator propagator(&hub, &oevt, octx->addEntry(ok.getEntryCode()) );

    OpticksViz* viz = ok.isCompute() ? NULL : new OpticksViz(&hub, &idx, true) ;


    FabStepNPY* fab = gen->makeFabstep();
    NPY<float>* gs = fab->getNPY();
    bool compute = false ; 
    gs->setBufferSpec(OpticksEvent::GenstepSpec(compute));  



    int multi = ok.getMultiEvent();

    for(int i=0 ; i < multi ; i++)
    {
        hub.createEvent(i);

        OpticksEvent* evt = hub.getEvent();
        evt->addBufferControl("seed", "VERBOSE_MODE");
        evt->addBufferControl("photon", "VERBOSE_MODE");

        evt->setGenstepData(gs);

        if(viz) viz->uploadEvent();           // via Scene and Rdr (only interop buffers)
        oevt.upload();                        // uploads gensteps, creates buffers at 1st upload, resizes on subsequent uploads

        propagator.prelaunch();               // keeps Thrust and OptiX talking about the same buffers

        seeder.seedPhotonsFromGensteps() ;    // Thrust: seed photon buffer (or WITH_SEED_BUFFER the seed buffer) using the genstep numPhotons for each step
        oevt.markDirty();                     // inform OptiX that must sync up input buffers with ctrl BUFFER_COPY_ON_DIRTY

        propagator.launch();                  // write the photon, record and sequence buffers


        if(viz) viz->downloadEvent();
        oevt.downloadPhotonData();            // allocates hostside buffer and copies into it (WILL SKIP IN INTEROP)


        NPY<float>* photon = evt->getPhotonData();
        const char* photon_path = "$TMP/VizTest_photon.npy";
        photon->save(photon_path);
        SSys::npdump(photon_path, "np.int32");


    }


    if(viz) viz->visualize(); 

    return 0 ;     
}

/*

    VizTest --compute    
       # this gives random bits for the photons (expected as incomplete fabricated gensteps), 
       # seed buffer appears correct

    VizTest  --compute --nopropagate
       # all zeros in photon buffer, seeds correct 

    VizTest  --compute --trivial 
       # expected debug quad2 and quad3, gensteps referenced correctly

    VizTest  --compute --trivial --multievent 2
    VizTest  --compute --trivial --multievent 10
       # all evt getting seeded correctly 

    VizTest --compute --dumpseed
       # expected last quad indices

    VizTest --dumpseed
       # phew: interop running that doesnt crash
       # seeding looks correct, rest of photon buffer has random OpenGL buffer bits

    VizTest --trivial
       # expected buffer debug entries in photon buffer

    VizTest --trivial --multievent 2
       # 1st event correct as above
       # 2nd event seed buffer correct but photon buffer all zeros 

    OKTest 
       # works, visualization and index operational 


*/



