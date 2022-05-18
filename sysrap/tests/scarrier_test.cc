/**
scarrier_test.cc : CPU tests of scarrier.h CUDA code using mocking 
================================================================

Standalone compile and run with::

   ./scarrier_test.sh 

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

const char* FOLD = SPath::Resolve("$TMP/scarrier_test", DIRPATH) ; 

NP* make_carrier_photon( const NP* gs, const NP* se )
{
    const quad6* gg = (quad6*)gs->bytes() ;  
    const int*   seed = (int*)se->bytes() ;  

    curandStateXORWOW rng(1u); 

    int tot_photon = se->shape[0] ; 
    NP* ph = NP::Make<float>( tot_photon, 4, 4); 
    quad4* pp = (quad4*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 

        quad4& p = pp[photon_id] ; 
        const quad6& g = gg[genstep_id] ;  
        
        scarrier::generate(p, rng, g, photon_id, genstep_id ); 

        std::cout << p.desc() << std::endl;  
    }
    return ph ; 
}

void test_generate()
{
    NP* gs = SEvent::MakeCarrierGensteps(); 
    NP* se = SEvent::MakeSeed(gs) ; 
    NP* ph = make_carrier_photon(gs, se); 

    printf("save to %s\n", FOLD );
    gs->save(FOLD, "gs.npy"); 
    se->save(FOLD, "se.npy"); 
    ph->save(FOLD, "ph.npy"); 
}

int main(int argc, char** argv)
{
    test_generate(); 
    return 0 ; 
}

