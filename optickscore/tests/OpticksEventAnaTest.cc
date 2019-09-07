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
OpticksEventAnaTest
=====================

Pulling together an evt and the NCSGList geometry 
it came from, for intersect tests.

**/

#include "NCSGList.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"
#include "OpticksEventAna.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure();

    OpticksEvent* evt = ok.loadEvent(true);
    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load ok evt " ; 
        return 0 ; 
    }
    const char* geopath = evt->getGeoPath();
    LOG(info) << " geopath : " << ( geopath ? geopath : "-" ) ; 

    if( geopath == NULL )
    {
        LOG(fatal) << "suceeded to load ok evt, BUT it has no associated geopath " ; 
        return 0 ; 
    }

    NCSGList* csglist = NCSGList::Load(geopath, ok.getVerbosity() );
    csglist->dump();

    OpticksEventAna*  okana = new OpticksEventAna(&ok, evt, csglist);
    okana->dump("GGeoTest::anaEvent.ok");
   

    OpticksEvent* g4evt = ok.loadEvent(false);
    if(!g4evt || g4evt->isNoLoad())
    {
        LOG(fatal) << "failed to load g4 evt " ; 
        return 0 ; 
    }
 
    const char* geopath2 = g4evt->getGeoPath();
    assert( strcmp( geopath, geopath2) == 0 );

    OpticksEventAna* g4ana = new OpticksEventAna(&ok, g4evt, csglist);
    g4ana->dump("GGeoTest::anaEvent.g4");

    if(okana) okana->dumpPointExcursions("ok");
    if(g4ana) g4ana->dumpPointExcursions("g4");

    return 0 ; 
}
