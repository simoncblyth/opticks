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

#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"
#include "OpticksCfg.hh"

#include "CG4.hh"
#include "CMaterialLib.hh"
#include "CTestDetector.hh"
#include "CTraverser.hh"

#include "G4VPhysicalVolume.hh"

#include "NBoundingBox.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv)

    LOG(info) << argv[0] ; 

    //const char* forced = "--test --apmtload " ;   // huh : why the --test ? that signifyies modify geometry 
    //  guess that the fail with the forced is because the default moified test bib geometry is not reversed
    const char* forced = NULL ; 

    Opticks ok(argc, argv, forced);
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksResource::SetupG4Environment(); 


    OpticksHub hub(&ok);

    CG4 g4(&hub);

    LOG(info) << "CG4 DONE" ; 
    CDetector* detector  = g4.getDetector();

    bool valid = detector->isValid();

    if(!valid)
    {
        LOG(error) << "Detector not valid " ;
        return 0 ; 
    } 



    detector->setVerbosity(2) ;

    CMaterialLib* mlib = detector->getMaterialLib() ;
    assert(mlib); 

    G4VPhysicalVolume* world_pv = detector->getTop();
    assert(world_pv);



    return 0 ; 
}
