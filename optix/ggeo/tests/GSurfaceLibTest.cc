// ggv --surf
// ggv --surf 6    // summary of all and detail of just the one index

#include "GCache.hh"
#include "GSurfaceLib.hh"

int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_", "surf.log", "info");
    gc.configure(argc, argv); 

    GSurfaceLib* lib = GSurfaceLib::load(&gc);
    lib->dump();

    return 0 ;
}

