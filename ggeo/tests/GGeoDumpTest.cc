#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"

int main(int argc, char** argv)
{
    int repeatIdx   = argc > 1 ? atoi(argv[1]) : -1 ; 
    int primIdx     = argc > 2 ? atoi(argv[2]) : -1 ; 
    int partIdxRel  = argc > 3 ? atoi(argv[3]) : -1 ; 

    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    GGeo* gg = GGeo::Load(&ok);

    gg->dumpParts("GGeoDumpTest.main", repeatIdx, primIdx, partIdxRel ) ;

    return 0 ;
}
