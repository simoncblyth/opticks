#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure(); 

    GGeo gg(&ok);
    gg.loadFromCache();
    gg.dumpStats();

    gg.dryrun_convert(); 

    return 0 ;
}


