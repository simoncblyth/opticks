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


    // TODO: relocate below preps into loadFromCache ? 
    gg.close();                  // normally OpticksHub::loadGeometry
    gg.deferredCreateGParts();   // normally OpticksHub::init 

  
    gg.dryrun_convert(); 

    return 0 ;
}


