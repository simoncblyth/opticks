#include "GCache.hh"
#include "GMaterialLib.hh"

int main()
{
    GCache gc("GGEOVIEW_");

    GMaterialLib* mlib = GMaterialLib::load(&gc);

    mlib->dump();

    return 0 ;
}

