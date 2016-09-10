#include "Opticks.hh"
#include "OpticksEvent.hh"

#include "TrivialCheckNPY.hpp"


#include "OKCORE_LOG.hh"
#include "GGV_LOG.hh"
#include "NPY_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ;  
    OKCORE_LOG__ ;  
    GGV_LOG__ ;  

    Opticks ok(argc, argv);
    ok.configure();

    int multi = m_ok->getMultiEvent();

    for(int tagoffset=0 ; tagoffset < multi ; tagoffset++)
    {
        OpticksEvent* evt = ok.loadEvent(true, tagoffset);  // ok=true, tagoffset=0
        evt->Summary();

        TrivialCheckNPY tcn(evt->getPhotonData(), evt->getGenstepData());
        tcn.check(argv[0]);
    }

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

   OKTest --cerenkov --trivial --save --compute
   TrivialTest --cerenkov 

For multievent::

   OKTest --cerenkov --trivial --save --compute --multievent 3 
   TrivialTest --cerenkov --multievent 3



**/

