/**
scarrier_test.cc : CPU tests of scarrier.h CUDA code using mocking 
================================================================

Standalone compile and run with::

   ~/o/sysrap/tests/scarrier_test.sh 

**/
#include <numeric>
#include <vector>

#include "scuda.h"
#include "squad.h"
#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
#include "sphoton.h"
#include "scarrier.h"

#include "SPath.hh"
#include "SEvent.hh"

#include "NP.hh"


NP* make_carrier_photon( const NP* gs, const NP* se )
{
    const quad6* gg = (quad6*)gs->bytes() ;  
    const int*   seed = (int*)se->bytes() ;  

    curandStateXORWOW rng ;   // typedef to srng by s_mock_curand.h 

    int tot_photon = se->shape[0] ; 
    NP* ph = NP::Make<float>( tot_photon, 4, 4); 
    sphoton* pp = (sphoton*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 
        const quad6& g = gg[genstep_id] ;  

        sphoton& p = pp[photon_id] ; 
 
        scarrier::generate(p, rng, g, photon_id, genstep_id ); 

        std::cout << p.desc() << std::endl;  
    }
    return ph ; 
}

void test_generate()
{
    NP* gs = SEvent::MakeCarrierGenstep(); 
    NP* se = SEvent::MakeSeed(gs) ; 
    NP* ph = make_carrier_photon(gs, se); 

    gs->save("$FOLD/gs.npy"); 
    se->save("$FOLD/se.npy"); 
    ph->save("$FOLD/ph.npy"); 
}

int main(int argc, char** argv)
{
    test_generate(); 
    return 0 ; 
}

