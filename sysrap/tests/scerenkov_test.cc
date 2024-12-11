/**
scerenkov_test.cc : CPU tests of scerenkov.h CUDA code using mocking 
========================================================================

Standalone compile and run with::

    ~/o/sysrap/tests/scerenkov_test.sh 

TODO: instead of this use actual qcerenkov.h together with srngcpu.h ?



**/
#include <numeric>
#include <vector>

#include "srngcpu.h"
using RNG = srngcpu ; 

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "scerenkov.h"

#include "SEvent.hh"

#include "NP.hh"

NP* make_cerenkov_photon( const NP* gs, const NP* se )
{
    const quad6* gg = (quad6*)gs->bytes() ;  
    const int*   seed = (int*)se->bytes() ;  

    RNG rng ; 
  
    int tot_photon = se->shape[0] ; 
    NP* ph = NP::Make<float>( tot_photon, 4, 4); 
    sphoton* pp = (sphoton*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 

        sphoton& p = pp[photon_id] ; 
        const quad6& g = gg[genstep_id] ;  
        
        // scerenkov::generate(p, rng, g, photon_id, genstep_id ); 
        // HUH : there isnt one 

        std::cout << p.desc() << std::endl;  
    }
    return ph ; 
}

void test_generate()
{
    NP* gs = SEvent::MakeCerenkovGenstep(); 
    NP* se = SEvent::MakeSeed(gs) ; 
    NP* ph = make_cerenkov_photon(gs, se); 

    gs->save("$FOLD/gs.npy"); 
    se->save("$FOLD/se.npy"); 
    ph->save("$FOLD/ph.npy"); 
}

int main(int argc, char** argv)
{
    test_generate(); 
    return 0 ; 
}

