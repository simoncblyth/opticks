#include "Types.hpp"
#include "GItemIndex.hh"
#include "stdlib.h"



void test_material_labelling()
{
    const char* idpath = getenv("IDPATH");
    const char* itemtype = "MaterialSequence" ; // seqmat from main

    GItemIndex* idx = GItemIndex::load(idpath, itemtype); 
    idx->dump();
 
    Types types ; 
    types.readMaterials(idpath, "GMaterialIndex");

    idx->setTypes(&types);
    idx->setLabeller(GItemIndex::MATERIALSEQ);

    idx->test();
    idx->dump();
    idx->formTable();
}






int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* itemtype = "FlagSequence" ; // seqhis from main

    GItemIndex* idx = GItemIndex::load(idpath, itemtype); 
    idx->dump();
 
    Types types ; 
    //types.readMaterials(idpath, "GMaterialIndex");
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");

    idx->setTypes(&types);
    idx->setLabeller(GItemIndex::HISTORYSEQ);

    idx->test();
    idx->dump();
    idx->formTable();

    return 0 ;
}
