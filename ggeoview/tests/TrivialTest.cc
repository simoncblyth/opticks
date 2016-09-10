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

    int multi = ok.getMultiEvent();

    int fail(0);

    for(int tagoffset=0 ; tagoffset < multi ; tagoffset++)
    {
        LOG(fatal) << " ################## tagoffset " << tagoffset ; 

        OpticksEvent* evt = ok.loadEvent(true, tagoffset);  // ok=true, tagoffset=0
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

   OKTest --compute --cerenkov --save --trivial
   TrivialTest --cerenkov 

For multievent (1 is default anyhow so this is same as above)::

   OKTest --cerenkov --trivial --save --compute --multievent 1
   TrivialTest --cerenkov --multievent 1

Up to 2::

   OKTest --cerenkov --trivial --save --compute --multievent 2
   TrivialTest --cerenkov    ## running this now fails with values stuck on first genstep
   TrivialTest --cerenkov --multievent 2



The seeds (genstep_id) stuck at zero in multi-event::

    In [9]: ox.view(np.int32)
    Out[9]: 
    array([[[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [         0,          0,          0,          0]],

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [         1,          4,          0,          0]],

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [         2,          8,          0,          0]],

           ..., 
           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [    612838,    2451352,          0,          0]],

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [    612839,    2451356,          0,          0]],

           [[         0,          0,          0,          0],
            [1065353216, 1065353216, 1065353216, 1065353216],
            [         1,          1,         67,         80],
            [    612840,    2451360,          0,          0]]], dtype=int32)

    In [10]: len(ox)
    Out[10]: 612841

    In [11]: len(ox)*4
    Out[11]: 2451364




**/

