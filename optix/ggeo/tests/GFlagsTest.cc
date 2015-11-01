#include "GCache.hh"
#include "GFlags.hh"
#include "GAttrSeq.hh"

int main()
{
    GCache gc("GGEOVIEW_");

    GFlags gf(&gc);

    GAttrSeq* q = gf.getAttrIndex(); 

    q->dump();

    return 0 ; 
}
