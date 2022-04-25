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
#include "storch.h"

#include "SEvent.hh"

#include "NP.hh"

const char* FOLD = "/tmp/storch_test" ; 


/*
void fill_torch_genstep( torch& gs, unsigned genstep_id, unsigned numphoton_per_genstep )
{
   // TODO: string configured gensteps, rather the the currently fixed and duplicated one  
    float3 mom = make_float3( 1.f, 1.f, 1.f ); 
    gs.wavelength = 501.f ; 
    gs.mom = normalize(mom); 
    gs.radius = 100.f ; 
    gs.pos = make_float3( 1000.f, 1000.f, 1000.f ); 
    gs.time = 0.f ; 
    gs.zenith = make_float2( 0.f, 1.f ); 
    gs.azimuth = make_float2( 0.f, 1.f ); 
    gs.type = storchtype::Type("disc");  
    gs.mode = 255 ;    //torchmode::Type("...");  
    gs.numphoton = numphoton_per_genstep  ; 
}

NP* make_torch_gs(unsigned num_gs, unsigned numphoton_per_genstep )
{
    NP* gs = NP::Make<float>(num_gs, 6, 4 ); 
    torch* tt = (torch*)gs->bytes() ; 
    for(unsigned i=0 ; i < num_gs ; i++ ) fill_torch_genstep( tt[i], i, numphoton_per_genstep ) ; 
    return gs ;  
}

*/


NP* make_seed(const NP* gs)
{
    assert( gs->has_shape(-1,6,4) ); 
    int num_gs = gs->shape[0] ; 
    const torch* tt = (torch*)gs->bytes() ; 

    std::vector<int> gsp(num_gs) ; 
    for(int i=0 ; i < num_gs ; i++ ) gsp[i] = tt[i].numphoton ;

    int tot_photons = 0 ; 
    for(int i=0 ; i < num_gs ; i++ ) tot_photons += gsp[i] ; 
    printf("//tot_photons %d \n", tot_photons ) ; 

    NP* se = NP::Make<int>( tot_photons ); 
    int* sev = se->values<int>();  

    int offset = 0 ; 
    for(int i=0 ; i < num_gs ; i++) 
    {   
        int np = gsp[i] ; 
        for(int p=0 ; p < np ; p++) sev[offset+p] = i ; 
        offset += np ; 
    }   
    return se ; 
}   

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
    //unsigned num_gs = 10 ; 
    //unsigned numphoton_per_gs = 100 ; 
    //NP* gs = make_torch_gs(num_gs, numphoton_per_gs ) ; 

    NP* gs = SEvent::MakeTorchGensteps(); 
    NP* se = make_seed(gs) ; 
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

        torch& t = qt.t ;  // when going down from union type to subtype can just use the union member without casting
        std::cout <<  qt.desc() << std::endl ; 
    }

    {
        quad6 gs ; 
        gs.zero(); 
        gs.q0.u.x = 202 ; 

        torch& t = (torch&)gs ;   // bolshy : simply cast across from one of the union-ed types to the other 

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

