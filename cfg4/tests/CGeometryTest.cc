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

#include <cassert>

#include "CFG4_BODY.hh"

// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

//#include "GGeo.hh"
#include "GGeoBase.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"

// cfg4-
#include "CG4.hh"
#include "CGeometry.hh"
#include "CDetector.hh"
#include "CMaterialTable.hh"

//#include "CBorderSurfaceTable.hh"
//#include "CSkinSurfaceTable.hh"

#include "CMaterialBridge.hh"
#include "CSurfaceBridge.hh"

// g4-
#include "G4VPhysicalVolume.hh"


#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    CFG4_LOG__ ; 
    GGEO_LOG__ ; 

    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);
    OpticksHub hub(&ok) ;

    CSensitiveDetector* sd = NULL ; 
    CGeometry cg(&hub, sd);

    CDetector* detector = cg.getDetector();

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);

    //detector->dumpLV();


/*
    CMaterialTable mt ; 
    mt.dump("CGeometryTest CMaterialTable");
    mt.dumpMaterial("GdDopedLS");
*/

    // these now moved inside CSurfaceBridge
    //CBorderSurfaceTable bst ; 
    //bst.dump("CGeometryTest CBorderSurfaceTable");
    //CSkinSurfaceTable sst ; 
    //sst.dump("CGeometryTest CSkinSurfaceTable");

   
    GGeoBase* ggb = hub.getGGeoBase();
    GBndLib* blib = ggb->getBndLib(); 
    GMaterialLib* mlib = blib->getMaterialLib(); 
    GSurfaceLib*  slib = blib->getSurfaceLib(); 

    CMaterialBridge mbr(mlib) ;
    mbr.dump("CGeometryTest"); 

    CSurfaceBridge sbr(slib);
    sbr.dump("CGeometryTest CSurfaceBridge");

    return 0 ; 
}
