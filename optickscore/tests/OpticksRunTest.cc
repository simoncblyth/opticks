#include <cassert>
#include <iostream>

#include "SProc.hh"

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

void test_OpticksRun_reset(Opticks* ok, unsigned nevt, char ctrl)
{
    unsigned num_photons = 10000 ; 
    NPY<float>* gs0 = OpticksGenstep::MakeCandle(num_photons, 0); 

    float vm0 = SProc::VirtualMemoryUsageMB() ; 

    for(unsigned i=0 ; i < nevt ; i++)
    {
        LOG(info) << i ; 
        gs0->setArrayContentIndex(i); 
        ok->createEvent(gs0, ctrl);   // input argument gensteps are cloned by OpticksEvent 
        ok->resetEvent(ctrl); 
    }

    float vm1 = SProc::VirtualMemoryUsageMB() ; 
    float dvm = vm1 - vm0 ; 
    float leak_per_evt = dvm/float(nevt) ; 

    LOG(info) 
       << " vm0 " << vm0
       << " vm1 " << vm1
       << " dvm " << dvm
       << " nevt " << nevt 
       << " leak_per_evt (MB) " << leak_per_evt 
       << " ctrl [" << ctrl << "]" 
       ;

    ok->saveProfile(); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int nevt = argc > 1 ? atoi(argv[1]) : 1000 ; 
    Opticks ok(argc, argv); 
    ok.configure(); 

    glm::vec4 space_domain(0.f,0.f,0.f,1000.f); 
    ok.setSpaceDomain(space_domain); 

    char ctrl = '-' ; 
    test_OpticksRun_reset(&ok, nevt, ctrl ); 

    return 0 ;
}

