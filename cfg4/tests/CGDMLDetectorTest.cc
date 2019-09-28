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

// TEST=CGDMLDetectorTest om-t

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
#include "CSkinSurfaceTable.hh"

// g4-
#include "G4VPhysicalVolume.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "OPTICKS_LOG.hh"

const char* TMPDIR = "$TMP/cfg4/CGDMLDetectorTest" ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);

    OpticksHub hub(&ok);

    LOG(error) << "//////////////////////////  AFTER OpticksHub instanciation ///////////////////////////////////// " ; 
 

    //OpticksCfg<Opticks>* m_cfg = m_opticks->getCfg();
    //m_cfg->commandline(argc, argv);  


    OpticksQuery* query = ok.getQuery();   // non-done inside Detector classes for transparent control/flexibility 

    CSensitiveDetector* sd = NULL ; 

    CGDMLDetector* detector  = new CGDMLDetector(&hub, query, sd) ; 

    LOG(error) << "//////////////////////////  AFTER CGDMLDetector instanciation ///////////////////////////////////// " ; 

    ok.setIdPathOverride(TMPDIR);
    detector->saveBuffers();
    ok.setIdPathOverride(NULL);

    bool valid = detector->isValid();
    if(!valid)
    {
        LOG(error) << "CGDMLDetector not valid " ;
        return 0 ;  
    }


    detector->setVerbosity(2) ;

    NPY<float>* gtransforms = detector->getGlobalTransforms();
    gtransforms->save(TMPDIR, "gdml.npy");

    unsigned int index = 3160 ; 

    glm::mat4 mg = detector->getGlobalTransform(index);
    glm::mat4 ml = detector->getLocalTransform(index);

    LOG(info) << " index " << index 
              << " pvname " << detector->getPVName(index) 
              << " global " << gformat(mg)
              << " local "  << gformat(ml)
              ;

    G4VPhysicalVolume* world_pv = detector->Construct();
    assert(world_pv);

    CMaterialTable mt ; 
    mt.dump("CGDMLDetectorTest CMaterialTable");

    CBorderSurfaceTable bst ; 
    bst.dump("CGDMLDetectorTest CBorderSurfaceTable");

    CSkinSurfaceTable skt ; 
    skt.dump("CGDMLDetectorTest CSkinSurfaceTable");

    return 0 ; 
}
