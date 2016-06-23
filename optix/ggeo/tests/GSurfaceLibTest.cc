// ggv --surf
// ggv --surf 6        // summary of all and detail of just the one index
// ggv --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"
#include "GSurfaceLib.hh"
#include "GGEO_CC.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks* opticks = new Opticks(argc, argv, "surf.log");

    GSurfaceLib* slib = GSurfaceLib::load(opticks);

    slib->dump();

    return 0 ;
}

