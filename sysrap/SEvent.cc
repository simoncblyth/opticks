#include "NP.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "sframe.h"

#include "storch.h"
#include "scerenkov.h"
#include "sscint.h"
#include "scarrier.h"

#include "sxyz.h" 


#include "NP.hh"
#include "PLOG.hh"
#include "SStr.hh"
#include "SSys.hh"

#include "OpticksGenstep.h"


#include "SGenstep.hh"
#include "SEvent.hh"

const plog::Severity SEvent::LEVEL = PLOG::EnvLevel("SEvent", "DEBUG") ; 


NP* SEvent::MakeDemoGensteps(const char* config)
{
    NP* gs = nullptr ;
    if(     SStr::StartsWith(config, "count")) gs = MakeCountGensteps(config) ;   
    else if(SStr::StartsWith(config, "torch")) gs = MakeTorchGensteps() ; 
    assert(gs); 
    LOG(LEVEL) 
       << " config " << ( config ? config :  "-" )
       << " gs " << ( gs ? gs->sstr() : "-" )
       ;

    return gs ; 
}



NP* SEvent::MakeTorchGensteps(){    return MakeGensteps( OpticksGenstep_TORCH ) ; }
NP* SEvent::MakeCerenkovGensteps(){ return MakeGensteps( OpticksGenstep_CERENKOV ) ; }
NP* SEvent::MakeScintGensteps(){    return MakeGensteps( OpticksGenstep_SCINTILLATION ) ; }
NP* SEvent::MakeCarrierGensteps(){  return MakeGensteps( OpticksGenstep_CARRIER ) ; }

NP* SEvent::MakeGensteps( int gentype )
{
    unsigned num_gs = 1 ; 
    NP* gs = NP::Make<float>(num_gs, 6, 4 );  
    switch(gentype)
    {
        case  OpticksGenstep_TORCH:         FillGensteps<storch>(   gs, 100) ; break ; 
        case  OpticksGenstep_CERENKOV:      FillGensteps<scerenkov>(gs, 100) ; break ; 
        case  OpticksGenstep_SCINTILLATION: FillGensteps<sscint>(   gs, 100) ; break ; 
        case  OpticksGenstep_CARRIER:       FillGensteps<scarrier>( gs, 10)  ; break ; 
    }
    return gs ; 
}

template<typename T>
void SEvent::FillGensteps( NP* gs, unsigned numphoton_per_genstep )
{
    T* tt = (T*)gs->bytes() ; 
    for(int i=0 ; i < gs->shape[0] ; i++ ) T::FillGenstep( tt[i], i, numphoton_per_genstep ) ; 
}

template void SEvent::FillGensteps<storch>(    NP* gs, unsigned numphoton_per_genstep ); 
template void SEvent::FillGensteps<scerenkov>( NP* gs, unsigned numphoton_per_genstep ); 
template void SEvent::FillGensteps<sscint>(    NP* gs, unsigned numphoton_per_genstep ); 
template void SEvent::FillGensteps<scarrier>(  NP* gs, unsigned numphoton_per_genstep ); 


/**
SEvent::MakeSeed
------------------

Normally this is done on device using involved thrust, 
here is a simple CPU implementation of that.
 
**/

NP* SEvent::MakeSeed( const NP* gs )
{
    assert( gs->has_shape(-1,6,4) );  
    int num_gs = gs->shape[0] ; 
    const storch* tt = (storch*)gs->bytes() ; 

    std::vector<int> gsp(num_gs) ; 
    for(int i=0 ; i < num_gs ; i++ ) gsp[i] = tt[i].numphoton ;

    int tot_photons = 0 ; 
    for(int i=0 ; i < num_gs ; i++ ) tot_photons += gsp[i] ; 

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





NP* SEvent::MakeCountGensteps(const char* config, int* total ) // static 
{
    std::vector<int>* photon_counts_per_genstep = nullptr ; 
    if( config == nullptr )
    {
        (*photon_counts_per_genstep) = { 3, 5, 2, 0, 1, 3, 4, 2, 4 }; 
    }
    return MakeCountGensteps(*photon_counts_per_genstep, total);
}

/**
SEvent::MakeCountGensteps
---------------------------

Used by qudarap/tests/QEventTest.cc

**/


NP* SEvent::MakeCountGensteps(const std::vector<int>& counts, int* total ) // static 
{
    int gencode = OpticksGenstep_TORCH ;
    std::vector<quad6> gensteps ;

    if(total) *total = 0 ; 
    for(unsigned i=0 ; i < counts.size() ; i++)
    {
        quad6 gs ; gs.zero(); 

        int gridaxes = XYZ ;  
        int gsid = 0 ;  
        int photons_per_genstep = counts[i]; 

        if(total) *total += photons_per_genstep ; 

        SGenstep::ConfigureGenstep(gs, gencode, gridaxes, gsid, photons_per_genstep ); 

        gs.q1.f.x = 0.f ;  gs.q1.f.y = 0.f ;  gs.q1.f.z = 0.f ;  gs.q1.f.w = 0.f ;

        // identity transform to avoid nan 
        gs.q2.f.x = 1.f ;  gs.q2.f.y = 0.f ;  gs.q2.f.z = 0.f ;  gs.q2.f.w = 0.f ;
        gs.q3.f.x = 0.f ;  gs.q3.f.y = 1.f ;  gs.q3.f.z = 0.f ;  gs.q3.f.w = 0.f ;
        gs.q4.f.x = 0.f ;  gs.q4.f.y = 0.f ;  gs.q4.f.z = 1.f ;  gs.q4.f.w = 0.f ;
        gs.q5.f.x = 0.f ;  gs.q5.f.y = 0.f ;  gs.q5.f.z = 0.f ;  gs.q5.f.w = 1.f ;

        gensteps.push_back(gs);
    }
    return SGenstep::MakeArray(gensteps);
}

unsigned SEvent::SumCounts(const std::vector<int>& counts) // static 
{
    unsigned total = 0 ; 
    for(unsigned i=0 ; i < counts.size() ; i++) total += counts[i] ; 
    return total ; 
}


/**
SEvent::ExpectedSeeds
----------------------

From a vector of counts populate the vector of seeds by simple CPU side duplication.  

**/

void SEvent::ExpectedSeeds(std::vector<int>& seeds, const std::vector<int>& counts ) // static 
{
    unsigned total = SumCounts(counts);  
    unsigned ni = counts.size(); 
    for(unsigned i=0 ; i < ni ; i++)
    {   
        int np = counts[i] ; 
        for(int p=0 ; p < np ; p++) seeds.push_back(i) ; 
    }   
    assert( seeds.size() == total );  
}

int SEvent::CompareSeeds( const int* seeds, const int* xseeds, int num_seed ) // static 
{
    int mismatch = 0 ; 
    for(int i=0 ; i < num_seed ; i++) if( seeds[i] != xseeds[i] ) mismatch += 1 ; 
    return mismatch ; 
}


std::string SEvent::DescSeed( const int* seed, int num_seed, int edgeitems )  // static 
{
    std::stringstream ss ; 
    ss << "SEvent::DescSeed num_seed " << num_seed << " (" ;

    for(int i=0 ; i < num_seed ; i++)
    {   
        if( i < edgeitems || i > num_seed - edgeitems ) ss << seed[i] << " " ; 
        else if( i == edgeitems )  ss << "... " ; 
    }   
    ss << ")"  ;   
    std::string s = ss.str(); 
    return s ; 
}
























