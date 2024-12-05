#pragma once
/**
SGenerate.h
=============

This exists for a very specific purpose : to enable
comparison between Opticks and Geant4 by providing 
a way for photons from some types of gensteps to 
be CPU generated in order to allow giving those
photons to Geant4. 


This started in SGenstep.hh but because it relies on 
the MOCK_CURAND macro the method was moved to this header only SGenerate.h  
to allow switching on MOCK_CURAND in the "user" code 
rather than the widely used sysrap library. 

**/


struct NP ; 
struct SGenerate
{
    static constexpr const char* EKEY = "SGenerate__GeneratePhotons_RNG_PRECOOKED" ; 
    static NP* GeneratePhotons(const NP* gs);  
}; 


#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "storch.h"
#include "scarrier.h"
#include "scurand.h"   // without MOCK_CURAND this is an empty struct only 
#include "SGenstep.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "OpticksGenstep.h"
#include "NP.hh"


/**
SGenerate::GeneratePhotons
----------------------------

Does high level genstep handling, prepares MOCK CURAND, 
creates seeds, creates photon array. 
The details of the generation are done by storch::generate or scarrier:generate

NB : currently only limited gentype can be generated with this

Q: Does MOCK_CURAND generate the same photons as without ? 
A: YES, see SGenerate__test.sh : MOCK_CURAND is to allow 
   code intended to run via CUDA to see GPU like API on the CPU 

**/

inline NP* SGenerate::GeneratePhotons(const NP* gs_ )
{
    bool rng_precooked = ssys::getenvbool(EKEY); 
    std::cerr 
        << "SGenerate::GeneratePhotons"
        << " " << EKEY 
        << " : "
        << ( rng_precooked ? "YES" : "NO " ) 
        << std::endl 
        ; 

    const quad6* gg = (quad6*)gs_->bytes() ; 
    NP* se = SEvent::MakeSeed(gs_) ;
    const int*   seed = (int*)se->bytes() ;   

    int tot_photon = se->shape[0] ; 
    NP* ph = NP::Make<float>( tot_photon, 4, 4); 
    sphoton* pp = (sphoton*)ph->bytes() ; 

    unsigned rng_seed = 1u ; 
#if defined(MOCK_CURAND)
    curandStateXORWOW rng(rng_seed); 
#else
    srng rng(rng_seed);  
#endif

    for(int i=0 ; i < tot_photon ; i++ )
    {   
        unsigned photon_id = i ; 
        unsigned genstep_id = seed[photon_id] ; 
        sphoton& p = pp[photon_id] ; 
        const quad6& gs = gg[genstep_id] ;   
        int gencode = SGenstep::GetGencode(gs);  

        if(rng_precooked) rng.setSequenceIndex(i);  
        switch(gencode)
        {    
            case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ; 
            case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ; 
            case OpticksGenstep_INPUT_PHOTON:    assert(0)  ; break ; 
        }    
        if(rng_precooked) rng.setSequenceIndex(-1);  
    }
    delete se ; 
    return ph ;
}


