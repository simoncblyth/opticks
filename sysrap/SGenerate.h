#pragma once
/**
SGenerate.h
=============

This started in SGenstep.hh but because it relies on 
the MOCK_CURAND macro the method was moved to this header only SGenerate.h  
to allow switching on MOCK_CURAND in the "user" code 
rather than the widely used sysrap library. 

**/
struct NP ; 
struct SGenerate
{
    static NP* GeneratePhotons();  
    static NP* GeneratePhotons(const NP* gs);  
}; 

#if defined(MOCK_CURAND)

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "storch.h"
#include "scarrier.h"
#include "scurand.h"

#include "SGenstep.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "OpticksGenstep.h"
#include "NP.hh"

#endif

NP* SGenerate::GeneratePhotons()
{
    NP* gs = SEvt::GetGenstep();  // user code needs to instanciate SEvt and AddGenstep 
    return GeneratePhotons(gs);  
}
NP* SGenerate::GeneratePhotons(const NP* gs_)
{
    NP* ph = nullptr ; 

#if defined(MOCK_CURAND)
    std::cout << " gs " << ( gs_ ? gs_->sstr() : "-" ) << std::endl ; 
    const quad6* gg = (quad6*)gs_->bytes() ; 

    NP* se = SEvent::MakeSeed(gs_) ;
    std::cout << " se " << ( se ? se->sstr() : "-" ) << std::endl ; 
    const int*   seed = (int*)se->bytes() ;   

    curandStateXORWOW rng(1u); 

    int tot_photon = se->shape[0] ; 
    ph = NP::Make<float>( tot_photon, 4, 4); 
    sphoton* pp = (sphoton*)ph->bytes() ; 

    for(int i=0 ; i < tot_photon ; i++ )
    {   
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 
        sphoton& p = pp[photon_id] ; 
        const quad6& gs = gg[genstep_id] ;   

        int gencode = SGenstep::GetGencode(gs);  
        switch(gencode)
        {    
            case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ; 
            case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ; 
        }    
    }
    std::cout << " ph " << ( ph ? ph->sstr() : "-" ) << std::endl ; 
#endif

    return ph ; 
}


