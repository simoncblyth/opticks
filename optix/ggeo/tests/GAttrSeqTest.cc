//  ggv --attr
#include "Opticks.hh"

#include "OpticksAttrSeq.hh"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "GMaterialLib.hh"
#include "GBndLib.hh"
#include "Index.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>


void test_history_sequence(Opticks* opticks)
{
    OpticksFlags* flags = opticks->getFlags();
    OpticksAttrSeq* qflg = flags->getAttrIndex();
    assert(qflg);
    qflg->dump();

    Index* seqhis = opticks->loadHistoryIndex(); 
    assert(seqhis);
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
    assert(seqmat);
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
    assert(boundaries);
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
    Opticks ok(argc, argv);

    test_history_sequence(&ok);
    test_material_sequence(&ok);
    test_material_dump(&ok);
    
    test_index_boundaries(&ok);
}
