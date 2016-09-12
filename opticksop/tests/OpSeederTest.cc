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
#include "OPropagator.hh"   
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

    OScene scene(&hub);
    OContext* octx = scene.getOContext();
    OEvent oevt(&hub, octx );
    OpSeeder  seeder(&hub, &oevt );
    OPropagator propagator(&hub, &oevt, octx->addEntry(ok.getEntryCode()) );

    GenstepNPY* fab = GenstepNPY::Fabricate(TORCH, 100, 1000 ); // genstep_type, num_step, num_photons_per_step
    NPY<float>* gs = fab->getNPY();
    const char* oac_label = "GS_TORCH" ; 


    int multi = ok.getMultiEvent();

    for(int i=0 ; i < multi ; i++)
    {
        hub.createEvent(i);

        OpticksEvent* evt = hub.getEvent();
        evt->addBufferControl("photon", "VERBOSE_MODE");
        evt->setGenstepData(gs, true, oac_label);

        oevt.upload();                        // uploads gensteps, creates buffers at 1st upload, resizes on subsequent uploads
        seeder.seedPhotonsFromGensteps() ;    // Thrust: seed photon buffer using the genstep numPhotons for each step
        oevt.markDirtyPhotonBuffer();         // inform OptiX that must sync up the photon buffer

        propagator.launch();                  // write the photon, record and sequence buffers

        oevt.downloadPhotonData();            // allocates hostsize photon buffer and copies into it 

        NPY<float>* photon = evt->getPhotonData();
        const char* path = "$TMP/OpSeederTest.npy";
        photon->save(path);
        SSys::npdump(path, "np.int32");
    }

    return 0 ;     
}

/*

    OpSeederTest --compute    
       # this gives all zeros for the photons, as the fabricated gensteps 
       # are not complete, they just have photon count and a fake materialline
       # so they will not generate photons that can hit geometry

    OpSeederTest  --compute --nopropagate
       # actual seeds at the head of the photons are visible
       # lack of real gensteps doesnt matter as propagation is skipped

    OpSeederTest  --compute --trivial --nopropagate
       # actual seeds at the head of the photons, same as above

    OpSeederTest  --compute --trivial
       # trivial debug lines are visible

    OpSeederTest  --compute --trivial --multievent 2
       # 2nd event has genstep id stuck at zero

    OpSeederTest  --compute --nopropagate --multievent 2
       # huh: unexpectedly the seeds ARE BEING SET FOR THE 2nd event (when the propagate launch is skipped altogether)

    OpSeederTest  --compute --nothing --multievent 2
       # try launching but running a do nothing program 
       # again the seeds ARE being set, but there is no access to the buffer
       # so OptiX may be being clever and skipping a buffer upload 

    OpSeederTest  --compute --dumpseed --multievent 2
       # just dumping the indices in quad3 (and not touching the quad0 location of the seed)
       # demonstrates again that genstep_id are being seeded correctly for the 
       # 2nd event (just like the 1st)
       #
       # problem is that the 2nd seeding doesnt overwrite the zeroing       
 

    OpSeederTest  --compute --trivial --multievent 2
       # after changing to use BUFFER_COPY_ON_DIRTY and invoking  oevt.markDirtyPhotonBuffer(); 
       # above multievent now appears to see the changed seeds
       # 
       # NOTE: have implemented an input only seed buffer but not yet using it 
        



*/



