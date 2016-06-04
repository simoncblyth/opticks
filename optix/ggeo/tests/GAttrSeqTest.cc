//  ggv --attr
#include "Opticks.hh"

#include "GCache.hh"
#include "GFlags.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"
#include "GAttrSeq.hh"
#include "Index.hpp"

#include <iostream>
#include <iomanip>


void test_history_sequence(Opticks* cache)
{
    GFlags* flags = cache->getFlags();
    GAttrSeq* qflg = flags->getAttrIndex();
    qflg->dump();

    Index* seqhis = Index::load(cache->getIdPath(), "History_Sequence");
    seqhis->dump();

    qflg->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
    qflg->dumpTable(seqhis, "seqhis"); 
}

void test_material_sequence(Opticks* cache)
{
    GMaterialLib* mlib = GMaterialLib::load(cache);
    GAttrSeq* qmat = mlib->getAttrNames();
    qmat->dump();

    Index* seqmat = Index::load(cache->getIdPath(), "Material_Sequence");
    seqmat->dump();

    qmat->setCtrl(GAttrSeq::SEQUENCE_DEFAULTS);
    qmat->dumpTable(seqmat, "seqmat"); 
}

void test_index_boundaries(Opticks* cache)
{
    GBndLib* blib = GBndLib::load(cache, true);
    blib->close(); 

    GAttrSeq* qbnd = blib->getAttrNames();
    qbnd->dump();

    Index* boundaries = Index::load(cache->getIdPath(), "indexBoundaries");
    boundaries->dump();
   

    qbnd->setCtrl(GAttrSeq::VALUE_DEFAULTS);
    qbnd->dumpTable(boundaries, "test_index_boundaries:dumpTable");
}


void test_material_dump(Opticks* cache)
{
    GMaterialLib* mlib = GMaterialLib::load(cache);
    GAttrSeq* qmat = mlib->getAttrNames();
    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;
    qmat->dump(mats);
}


int main(int argc, char** argv)
{
    Opticks ok(argc, argv);
    //GCache gc(&ok);

    test_history_sequence(&ok);
    test_material_sequence(&ok);
    test_material_dump(&ok);
    
    test_index_boundaries(&ok);
}
