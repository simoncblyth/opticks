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
#include <iostream>

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "RecordsNPY.hpp"
#include "OPTICKS_LOG.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEventSpec.hh"
#include "OpticksEvent.hh"
#include "OpticksGenstep.hh"


void test_genstep_derivative()
{
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

    NPY<float>* trk = evt.loadGenstepDerivativeFromFile("track");
    assert(trk);

    LOG(info) << trk->getShapeString();

    glm::vec4 origin    = trk->getQuadF(0,0) ;
    glm::vec4 direction = trk->getQuadF(0,1) ;
    glm::vec4 range     = trk->getQuadF(0,2) ;

    print(origin,"origin");
    print(direction,"direction");
    print(range,"range");

}


void test_genstep()
{   
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

/*
   not compiling anymore
    evt.setGenstepData(evt.loadGenstepFromFile());
    evt.dumpPhotonData();
*/
}

void test_appendNote()
{   
    OpticksEventSpec sp("cerenkov", "1", "dayabay", "") ;
    OpticksEvent evt(&sp) ;

    evt.appendNote("hello");
    evt.appendNote("world");

    std::string note = evt.getNote();

    bool match = note.compare("hello world") == 0 ;
    if(!match) LOG(fatal) << "got unexpected " << note ; 
    assert(match);

}

void test_resetEvent(Opticks* ok, bool cfg4evt)
{
    unsigned num_photons = 10000 ; 
    unsigned tagoffset = 0 ; 

    NPY<float>* gs = OpticksGenstep::MakeCandle(num_photons, tagoffset ) ;

    ok->createEvent(gs, cfg4evt); 
    delete gs ; 

    ok->resetEvent(); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    //test_genstep_derivative();
    //test_genstep();
    //test_appendNote();

    Opticks ok(argc, argv); 
    ok.configure(); 

    glm::vec4 space_domain(0.f,0.f,0.f,1000.f); 
    ok.setSpaceDomain(space_domain); 

    test_resetEvent(&ok, false); 
    test_resetEvent(&ok, true); 

    return 0 ;
}
