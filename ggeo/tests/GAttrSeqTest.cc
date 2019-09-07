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

//  op --attr
//
// To override the default geometry detector use "--cat" option eg::
//   
//     GAttrSeqTest --cat concentric
//

#include <iostream>
#include <iomanip>
#include <cassert>

#include "Index.hpp"

#include "Opticks.hh"
#include "OpticksAttrSeq.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "GMaterialLib.hh"
#include "GBndLib.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "OKCORE_LOG.hh"


void test_history_sequence(Opticks* opticks)
{
    //OpticksFlags* flags = opticks->getFlags();
    //OpticksAttrSeq* qflg = flags->getAttrIndex();
    OpticksAttrSeq* qflg = opticks->getFlagNames();

    assert(qflg);
    qflg->dump();

    Index* seqhis = opticks->loadHistoryIndex(); 
    if(!seqhis)
    {
        LOG(error) << "NULL seqhis" ;
        return ; 
    } 
    seqhis->dump();

    qflg->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);
    qflg->dumpTable(seqhis, "seqhis"); 
}

void test_material_sequence(Opticks* opticks)
{
    GMaterialLib* mlib = GMaterialLib::load(opticks);
    OpticksAttrSeq* qmat = mlib->getAttrNames();
    qmat->dump();

    Index* seqmat = opticks->loadMaterialIndex(); 
    if(!seqmat)
    {
        LOG(error) << "NULL seqmat" ;
        return ; 
    } 
    seqmat->dump();

    qmat->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);
    qmat->dumpTable(seqmat, "seqmat"); 
}

void test_index_boundaries(Opticks* opticks)
{
    GBndLib* blib = GBndLib::load(opticks, true);
    blib->close(); 

    OpticksAttrSeq* qbnd = blib->getAttrNames();
    qbnd->dump();

    Index* boundaries = opticks->loadBoundaryIndex(); 
    if(!boundaries)
    {
        LOG(error) << "NULL boundaries" ;
        return ; 
    } 
    boundaries->dump();

    qbnd->setCtrl(OpticksAttrSeq::VALUE_DEFAULTS);
    qbnd->dumpTable(boundaries, "test_index_boundaries:dumpTable");
}


void test_material_dump(Opticks* opticks)
{
    GMaterialLib* mlib = GMaterialLib::load(opticks);
    OpticksAttrSeq* qmat = mlib->getAttrNames();
    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;
    qmat->dump(mats);
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG__ ;
    OKCORE_LOG__ ;

    Opticks ok(argc, argv);
    ok.configure();

    test_history_sequence(&ok);
    test_material_sequence(&ok);
    test_material_dump(&ok);
    
    test_index_boundaries(&ok);
}
