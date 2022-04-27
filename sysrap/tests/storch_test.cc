/**
storch_test.cc : CPU tests of storch.h CUDA code using mocking 
================================================================

Standalone compile and run with::

   ./storch_test.sh 

**/
#include <numeric>
#include <vector>

#include "scuda.h"
#include "squad.h"
#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
#include "sphoton.h"
#include "storch.h"

#include "SEvent.hh"

#include "NP.hh"

const char* FOLD = "/tmp/storch_test" ; 

NP* make_torch_photon( const NP* gs, const NP* se )
{
    const quad6* gg = (quad6*)gs->bytes() ;  
    const int*   seed = (int*)se->bytes() ;  

    curandStateXORWOW rng(1u); 

    int tot_photon = se->shape[0] ; 
    NP* ph = NP::Make<float>( tot_photon, 4, 4); 
    qphoton* pp = (qphoton*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 

        qphoton& p = pp[photon_id] ; 
        const quad6& g = gg[genstep_id] ;  
        
        qtorch::generate(p, rng, g, photon_id, genstep_id ); 

        std::cout << p.p.desc() << std::endl;  
    }
    return ph ; 
}

void test_generate()
{
    NP* gs = SEvent::MakeTorchGensteps(); 
    NP* se = SEvent::MakeSeed(gs) ; 
    NP* ph = make_torch_photon(gs, se); 

    printf("save to %s\n", FOLD );
    gs->save(FOLD, "gs.npy"); 
    se->save(FOLD, "se.npy"); 
    ph->save(FOLD, "ph.npy"); 
}

void test_union_cast()
{
    {
        qtorch qt ; 
        qt.q.zero() ; 
        qt.q.q0.u.x = 101 ; 

        storch& t = qt.t ;  // when going down from union type to subtype can just use the union member without casting
        std::cout <<  qt.desc() << std::endl ; 
    }

    {
        quad6 gs ; 
        gs.zero(); 
        gs.q0.u.x = 202 ; 

        storch& t = (storch&)gs ;   // bolshy : simply cast across from one of the union-ed types to the other 

        std::cout <<  "t.gentype " << t.gentype << std::endl ; 
    }

    {
        quad6 gs ; 
        gs.zero(); 
        gs.q0.u.x = 303 ; 

        qtorch& qt = (qtorch&)gs ;   // simply cast from one of the union types up to the union type

        std::cout <<  qt.desc() << std::endl ; 
    }
}



int main(int argc, char** argv)
{
    test_generate(); 
    //test_union_cast(); 

    return 0 ; 
}

