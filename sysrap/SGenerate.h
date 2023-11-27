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
    //static NP* GeneratePhotons(int idx);  
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

NB only the below gentypes are implemented::

    INPUT_PHOTONS
    CARRIER
    TORCH 

Called for example from U4VPrimaryGenerator::GeneratePrimaries

inline NP* SGenerate::GeneratePhotons(int idx)
{
    NP* gs = SEvt::GatherGenstep(idx);  // LOOKS GENERAL, BUT ISNT 

   

    if( gs == nullptr )
    {
        std::cerr << "SGenerate::GeneratePhotons FATAL SEvt::GatherGenstep returns null " << std::endl ; 
        std::cerr << "user code needs to instanciate SEvt and AddGenstep " << std::endl ;  
        std::cerr << "NB even input photon running requires a genstep" << std::endl ;  
        return nullptr ;  
    }

    assert( gs->shape[0] == 1 );  // MAKING THE POINT THAT THIS CODE LOOKS MORE GENERAL THAN IT IS 


    NP* ph = nullptr ; 
    if(OpticksGenstep_::IsInputPhoton(SGenstep::GetGencode(gs,0)))
    {
        ph = SEvt::GetInputPhoton(idx); 
    }
    else
    {
        ph = GeneratePhotons(gs);
    }

    // HMM: should this selection be done here OR later 
    int rerun_id = SEventConfig::G4StateRerun() ;
    NP* phs = rerun_id > -1 ? NP::MakeSelectCopy(ph, rerun_id ) : NP::MakeCopy(ph) ;
    // NB: the array now carries idlist metadata with the rerun_id 

    if(rerun_id > -1 )
    {
        std::cout 
            << "SGenerate::GeneratePhotons"
            << " rerun_id " << rerun_id
            << " ph " << ( ph ? ph->brief() : "-" ) 
            << " phs " << ( phs ? phs->brief() : "-" ) 
            << " (" << SEventConfig::kG4StateRerun << ") " 
            << std::endl 
            ; 
    }

    return phs ;  
}
**/




/**
SGenerate::GeneratePhotons
----------------------------

Does high level genstep handling, prepares MOCK CURAND, 
creates seeds, creates photon array. 
The details of the generation are done by storch::generate or scarrier:generate

NB : currently only limited gentype can be generated with this

**/

inline NP* SGenerate::GeneratePhotons(const NP* gs_)
{
    //std::cout << "SGenerate::GeneratePhotons gs_ " <<  ( gs_ ? gs_->sstr() : "-" ) << std::endl ; 

    NP* ph = nullptr ; 

    const quad6* gg = (quad6*)gs_->bytes() ; 

    NP* se = SEvent::MakeSeed(gs_) ;
    //std::cout << " se " << ( se ? se->sstr() : "-" ) << std::endl ; 
    const int*   seed = (int*)se->bytes() ;   

    int tot_photon = se->shape[0] ; 
    ph = NP::Make<float>( tot_photon, 4, 4); 
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

        switch(gencode)
        {    
            case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ; 
            case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ; 
            case OpticksGenstep_INPUT_PHOTON:    assert(0)  ; break ; 
        }    
    }
    return ph ;
}


