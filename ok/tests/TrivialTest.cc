#include "Opticks.hh"
#include "OpticksEvent.hh"

#include "TrivialCheckNPY.hpp"


#include "OKCORE_LOG.hh"
#include "OK_LOG.hh"
#include "NPY_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ;  
    OKCORE_LOG__ ;  
    OK_LOG__ ;  

    Opticks ok(argc, argv);
    ok.configure();

    int multi = ok.getMultiEvent();

    LOG(info) << argv[0] 
              << " multi " << multi 
              ; 

    int fail(0);

    for(int tagoffset=0 ; tagoffset < multi ; tagoffset++)
    {
        LOG(fatal) << " ################## tagoffset " << tagoffset ; 

        OpticksEvent* evt = ok.loadEvent(true, tagoffset);  

        if(evt->isNoLoad()) 
        {
            LOG(error) << "FAILED to load evt from " << evt->getDir() ;
            continue ;  
        }
         

        evt->Summary();

        TrivialCheckNPY tcn(evt->getPhotonData(), evt->getGenstepData());
        fail += tcn.check(argv[0]);
    }

    LOG(info) << " fails: " << fail ; 
    assert(fail == 0);
 
    return 0 ; 
}

/**

TrivialTest
=============

Checks correspondence between input gensteps and the photon
buffer output by the trivial entry point, which does
minimal processing just checking that genstep seeds are
correctly uploaded and available at photon level in the OptiX 
program.

Produce single event to examine and check it with::

   OKTest --compute --save --trivial     ## default to torch 
   TrivialTest  

   OKTest --compute --cerenkov --save --trivial
   TrivialTest --cerenkov


   


For multievent (1 is default anyhow so this is same as above)::

   OKTest --cerenkov --trivial --save --compute --multievent 1
   TrivialTest --cerenkov --multievent 1


See :doc:`notes/issues/geant4_opticks_integration/multi_event_seed_stuck_at_zero_for_second_event`

   

**/

