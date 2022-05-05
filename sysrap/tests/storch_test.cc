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
    sphoton* pp = (sphoton*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 

        sphoton& p = pp[photon_id] ; 
        const quad6& g = gg[genstep_id] ;  
        
        storch::generate(p, rng, g, photon_id, genstep_id ); 

        std::cout << p.desc() << std::endl;  
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
        std::cout <<  t.desc() << std::endl ; 
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

        storch& t = (storch&)gs ;   // simply casting between union types

        std::cout <<  t.desc() << std::endl ; 
    }
}

void test_ref_cast()
{
    float4 a = make_float4( 1.f, 2.f, 3.f, 4.f ); 
    const float3& b = (const float3&)a ; 
    const float2& c = (const float2&)a ; 
    const float3& d = (const float3&)a.x ; 
    std::cout << " a " << a << std::endl ;   
    std::cout << " b " << b << std::endl ;   
    std::cout << " c " << c << std::endl ;   
    std::cout << " d " << d << std::endl ;   
}

int main(int argc, char** argv)
{
    test_generate(); 
    //test_union_cast(); 
    //test_ref_cast(); 
    return 0 ; 
}

