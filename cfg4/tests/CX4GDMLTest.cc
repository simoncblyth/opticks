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

//  cfg4-;cfg4-c;om-;TEST=CX4GDMLTest om-t
//  cfg4-;cfg4-c;om-;TEST=CX4GDMLTest om-d
//
//  while CX4GDMLTest ; do sleep 0.2 ; done 

#include <cassert>
// cfg4--;op --cgdmldetector --dbg

#include "CFG4_BODY.hh"

#include "Opticks.hh"     // okc-
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh"   // okg-

// cfg4-
#include "CTestDetector.hh"
#include "CGDMLDetector.hh"
#include "CMaterialTable.hh"
#include "CBorderSurfaceTable.hh"

// g4-
#include "G4VPhysicalVolume.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"


// x4-
#include "X4PhysicalVolume.hh"
#include "GGeo.hh"


#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);

    OpticksHub hub(&ok);


    OpticksQuery* query = ok.getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CSensitiveDetector* sd = NULL ; 

    CGDMLDetector* detector  = new CGDMLDetector(&hub, query, sd) ; 

    ok.setIdPathOverride("$TMP");
    detector->saveBuffers();
    ok.setIdPathOverride(NULL);

    bool valid = detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }

    detector->setVerbosity(2) ;

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);

    LOG(info) << "/////////////////////////////////////////////////" ;

   
    GGeo* ggeo = X4PhysicalVolume::Convert(world_pv, NULL) ;  // HMM: this will instanciate another Opticks ??
    assert( ggeo ); 

    ggeo->save();  // saves to /usr/local/opticks/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live

    LOG(info) << " DONE " ; 


    return 0 ; 
}
