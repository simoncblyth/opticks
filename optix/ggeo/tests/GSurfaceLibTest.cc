// ggv --surf
// ggv --surf 6        // summary of all and detail of just the one index
// ggv --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"
#include "GSurfaceLib.hh"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "surf.log");

    GSurfaceLib* slib = GSurfaceLib::load(opticks);

    slib->dump();

    return 0 ;
}

