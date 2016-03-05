// ggv --surf
// ggv --surf 6        // summary of all and detail of just the one index
// ggv --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"

#include "GCache.hh"
#include "GScintillatorLib.hh"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "scint.log");
    GCache gc(opticks);

    GScintillatorLib* slib = GScintillatorLib::load(&gc);
    slib->dump();

    return 0 ;
}

