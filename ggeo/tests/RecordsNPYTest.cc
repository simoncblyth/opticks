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

//  ggv --recs

#include <cassert>

#include "Types.hpp"
#include "Typ.hpp"
#include "RecordsNPY.hpp"
#include "PhotonsNPY.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "GGEO_BODY.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    LOG(info) << argv[0] ; 


    Opticks ok(argc, argv);

    ok.configure() ; // hub not available at ggeo- level 

    Types* types = ok.getTypes();
    GBndLib* blib = GBndLib::load(&ok, true); 
    GMaterialLib* mlib = blib->getMaterialLib();
    //GSurfaceLib*  slib = blib->getSurfaceLib();

    // see GGeo::setupTyp
    Typ* typ = ok.getTyp();
    typ->setMaterialNames(mlib->getNamesMap());

    //OpticksFlags* flags = ok->getFlags();
    //typ->setFlagNames(flags->getNamesMap());
    typ->setFlagNames(ok.getFlagNamesMap());


    OpticksEvent* evt = ok.loadEvent();
    if(evt->isNoLoad())
    {
        LOG(error) << "failed to load evt " 
                  // << evt->getDir() 
                   ;
        return 0 ;  
    }    

    NPY<float>* fd = evt->getFDomain();
    NPY<float>* ox = evt->getPhotonData();
    NPY<short>* rx = evt->getRecordData();


/*
    const char* idpath = cache->getIdPath();
    NPY<float>* fdom = NPY<float>::load(idpath, "OPropagatorF.npy");
    //NPY<int>*   idom = NPY<int>::load(idpath, "OPropagatorI.npy");

    // array([[[9, 0, 0, 0]]], dtype=int32)     ??? no 10 for maxrec 
    // NumpyEvt::load to do this ?

    NPY<float>* ox = NPY<float>::load("ox", src, tag, "dayabay");
    ox->Summary();

    NPY<short>* rx = NPY<short>::load("rx", src, tag, "dayabay");
    rx->Summary();

*/


    unsigned int maxrec = 10 ; 
    RecordsNPY* rec = new RecordsNPY(rx, maxrec);
    rec->setDomains(fd);
    rec->setTypes(types);
    rec->setTyp(typ);

    PhotonsNPY* pho = new PhotonsNPY(ox);
    pho->setRecs(rec);
    pho->setTypes(types);
    pho->setTyp(typ);
    pho->dump(0  ,  "ggv --recs dpho 0");

    return 0 ; 
}
