#include <cassert>
#include <iostream>

#include "NPY.hpp"
#include "OPTICKS_LOG.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksGenstep.hh"
#include "OpticksRun.hh"

/**

::

    TEST=OpticksRunTest ipython -i ~/opticks/ana/profile_.py -- --tag 0

    See notes/issues/OpticksRunTest_memory_reset_check.rst

**/

void test_OpticksRun_reset(Opticks* ok, unsigned nevt)
{
    unsigned num_photons = 10000 ; 
    NPY<float>* gs0 = OpticksGenstep::MakeCandle(num_photons, 0); 
    for(unsigned i=0 ; i < nevt ; i++)
    {
        LOG(info) << i ; 
        NPY<float>* gs = gs0->clone(); 
        gs->setArrayContentIndex(i); 
        bool cfg4evt = false ; 
        ok->createEvent(gs, cfg4evt); 
        ok->resetEvent(); 
    }
    ok->saveProfile(); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int nevt = argc > 1 ? atoi(argv[1]) : 10 ; 
    Opticks ok(argc, argv); 
    ok.configure(); 

    glm::vec4 space_domain(0.f,0.f,0.f,1000.f); 
    ok.setSpaceDomain(space_domain); 

    test_OpticksRun_reset(&ok, nevt ); 

    return 0 ;
}

