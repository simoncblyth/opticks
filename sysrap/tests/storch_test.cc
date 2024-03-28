/**
storch_test.cc : CPU tests of storch.h CUDA code using mocking 
================================================================

Standalone compile and run with::

   ~/opticks/sysrap/tests/storch_test.sh 

HMM: not standalone anymore, currently using libSysrap

**/
#include <numeric>
#include <vector>
#include <cstdlib>

#include "scuda.h"
#include "squad.h"
#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
#include "sphoton.h"
#include "storch.h"
#include "SEvent.hh"

#include "NPFold.h"


struct storch_test
{
    static void union_cast(); 
    static void ref_cast(); 
    static NP* make_torch_photon( const NP* gs, const NP* se );   
    static NPFold* generate();
};

void storch_test::union_cast()
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

void storch_test::ref_cast()
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






NP* storch_test::make_torch_photon( const NP* gs, const NP* se )
{
    std::cout << "[storch_test::make_torch_photon" << std::endl ;
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
         
        if(i % 100 == 0) std::cout << std::setw(6) << i << " : " << p.descBase() << std::endl;  
    }
    std::cout << "]storch_test::make_torch_photon" << std::endl ;
    return ph ; 
}

NPFold* storch_test::generate()
{
    std::cout << "[storch_test::generate" << std::endl ;
    NP* gs = SEvent::MakeTorchGenstep(); 
    NP* se = SEvent::MakeSeed(gs) ; 
    NP* ph = make_torch_photon(gs, se); 

    NPFold* fold = new NPFold ; 
    fold->add( "gs", gs ); 
    fold->add( "se", se ); 
    fold->add( "ph", ph ); 
    std::cout << "]storch_test::generate" << std::endl ;
    return fold ; 
}

int main(int argc, char** argv)
{
    /*
    storch_test::union_cast(); 
    storch_test::ref_cast(); 
    */

    NPFold* fold = storch_test::generate(); 
    fold->save("$FOLD"); 

    return 0 ; 
}

