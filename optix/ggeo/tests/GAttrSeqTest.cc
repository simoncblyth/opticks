//  ggv --attr

#include "GCache.hh"
#include "GFlags.hh"
#include "GMaterialLib.hh"
#include "GAttrSeq.hh"
#include "Index.hpp"

#include <iostream>
#include <iomanip>


void test_history_sequence(GCache* cache)
{
    GFlags* flags = cache->getFlags();
    GAttrSeq* qflg = flags->getAttrIndex();
    qflg->dump();

    Index* seqhis = Index::load(cache->getIdPath(), "History_Sequence");
    seqhis->dump();

    qflg->dumpHexTable(seqhis, "seqhis"); 
}

void test_material_sequence(GCache* cache)
{
    GMaterialLib* mlib = GMaterialLib::load(cache);
    GAttrSeq* qmat = mlib->getAttrNames();
    qmat->dump();

    Index* seqmat = Index::load(cache->getIdPath(), "Material_Sequence");
    seqmat->dump();

    qmat->dumpHexTable(seqmat, "seqmat"); 
}

void test_material_dump(GCache* cache)
{
    GMaterialLib* mlib = GMaterialLib::load(cache);
    GAttrSeq* qmat = mlib->getAttrNames();
    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;
    qmat->dump(mats);
}


int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_", "attr.log");
    gc.configure(argc, argv);

    test_history_sequence(&gc);

    test_material_sequence(&gc);

    test_material_dump(&gc);

}
