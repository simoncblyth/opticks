// ggv --surf
// ggv --surf 6        // summary of all and detail of just the one index
// ggv --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"

#include "GCache.hh"
#include "GSurfaceLib.hh"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "surf.log");
    GCache gc(opticks);

    GSurfaceLib* slib = GSurfaceLib::load(&gc);
    slib->dump();

    return 0 ;
}

