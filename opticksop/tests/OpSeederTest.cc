#include <string>
#include <sstream>

#include "SSys.hh"      // sysrap-

#include "NPY.hpp"      // npy-
#include "GenstepNPY.hpp"

#include "Opticks.hh"   // okc-
#include "OpticksPhoton.h"   
#include "OpticksEvent.hh"

#include "OpticksHub.hh"  // okg-
#include "OpticksRun.hh"  

#include "OContext.hh"   // optixrap-
#include "OScene.hh" 
#include "OEvent.hh" 

#include "OpSeeder.hh"   // opop-

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"

#include "PLOG.hh"

/**
OpSeederTest
============

**/

int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 
    OXRAP_LOG__ ; 
    OKOP_LOG__ ; 

    Opticks ok(argc, argv);

    OpticksHub hub(&ok);

    hub.createEvent(0);

    OpticksEvent* evt = hub.getEvent();


    unsigned genstep_type = TORCH ; 
    unsigned num_step = 100 ; 
    GenstepNPY* fab = GenstepNPY::Fabricate(genstep_type, num_step );
    NPY<float>* gs = fab->getNPY();

    const char* oac_label = "GS_TORCH" ; 
    evt->setGenstepData(gs, true, oac_label);


    OScene scene(&hub);

    OEvent oevt(&hub, scene.getOContext() );

    OpSeeder  seeder(&hub, &oevt );



    oevt.upload();

    seeder.seedPhotonsFromGensteps() ;

    oevt.downloadPhotonData();


    NPY<float>* photon = evt->getPhotonData();
    const char* path = "$TMP/OpSeederTest.npy";
    photon->save(path);
    SSys::npdump(path, "np.int32");

    return 0 ;     
}

/*

    OpSeederTest --compute 

*/



