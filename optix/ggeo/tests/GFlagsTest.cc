#include "GCache.hh"
#include "GFlags.hh"
#include "GAttrSeq.hh"
#include "Index.hpp"

int main()
{
    GCache gc("GGEOVIEW_");

    GFlags gf(&gc);

    GAttrSeq* q = gf.getAttrIndex(); 

    q->dump();

    Index* idx = gf.getIndex();

    idx->setExt(".ini");

    idx->save(gc.getIdPath());



    return 0 ; 
}
