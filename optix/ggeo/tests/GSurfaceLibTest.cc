#include "GCache.hh"
#include "GSurfaceLib.hh"

int main()
{
    GCache gc("GGEOVIEW_");

    GSurfaceLib* lib = GSurfaceLib::load(&gc);

    lib->dump();

    return 0 ;
}

