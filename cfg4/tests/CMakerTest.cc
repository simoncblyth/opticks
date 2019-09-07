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

#include "SSys.hh"
#include "NCSG.hpp"
#include "NPrimitives.hpp"

#include "BOpticksResource.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "CMaker.hh"

#include "G4VPhysicalVolume.hh"
#include "G4Sphere.hh"

#include "OPTICKS_LOG.hh"



void test_load_csg(int argc, char** argv)
{
    bool testgeo(false); 
    BOpticksResource okr(testgeo) ;  
    std::string treedir = okr.getDebuggingTreedir(argc, argv);

    NCSG* csg = NCSG::Load( treedir.c_str() );  
    if(!csg) return ; 

    csg->dump();

    G4VSolid* solid = CMaker::MakeSolid(csg);
    assert(solid); 
}

void test_make_csg()
{
    nsphere* sp = make_sphere();
    sp->set_boundary("Dummy"); 

    NCSG* csg = NCSG::Adopt(sp);
    if(!csg) return ; 
 
    csg->dump();

    G4VSolid* solid = CMaker::MakeSolid(csg);
    assert(solid); 

    G4Sphere* sp_ = dynamic_cast<G4Sphere*>(solid);
    assert(sp_);

    double radius = sp_->GetOuterRadius();
    assert( radius == 100. );

    LOG(info) << " sp " << *sp_ ; 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv)

    LOG(info) << argv[0] ; 

    //unsigned verbosity = SSys::getenvint("VERBOSITY", 1);

    Opticks ok(argc, argv);

    //test_load_csg(argc, argv);
    test_make_csg();


    return 0 ; 
} 
