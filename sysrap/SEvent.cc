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
#include "ssys.h"
#include "sstr.h"

#include "NPX.h"
#include "SLOG.hh"

#include "OpticksGenstep.h"

#include "SEvt.hh"
#include "SGenstep.hh"
#include "SEvent.hh"

const plog::Severity SEvent::LEVEL = SLOG::EnvLevel("SEvent", "DEBUG") ; 




NP* SEvent::GENSTEP = nullptr ; 
NP* SEvent::GetGENSTEP()
{
    LOG_IF(info, SEvt::LIFECYCLE ) <<  " GENSTEP " << ( GENSTEP ? GENSTEP->sstr() : "-" ); 
    return GENSTEP ; 
}

void SEvent::SetGENSTEP(NP* gs)
{
    GENSTEP = gs ; 
    LOG_IF(info, SEvt::LIFECYCLE ) <<  " GENSTEP " << ( GENSTEP ? GENSTEP->sstr() : "-" ); 
}

bool SEvent::HaveGENSTEP()
{
    return GENSTEP != nullptr ; 
}

NP* SEvent::HIT = nullptr ; 
NP* SEvent::GetHIT()
{
    return HIT ; 
}
void SEvent::SetHIT(NP* gs)
{
    HIT = gs ; 
}
bool SEvent::HaveHIT()
{
    return HIT != nullptr ; 
}










NP* SEvent::MakeDemoGenstep(const char* config, int idx)
{
    NP* gs = nullptr ;
    if(     sstr::StartsWith(config, "count")) gs = MakeCountGenstep(config) ;   
    else if(sstr::StartsWith(config, "torch")) gs = MakeTorchGenstep(idx) ; 
    assert(gs); 
    LOG(LEVEL) 
       << " config " << ( config ? config :  "-" )
       << " gs " << ( gs ? gs->sstr() : "-" )
       ;

    return gs ; 
}




NP* SEvent::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
{
    std::vector<quad6> qgs(1) ; 
    qgs[0].zero() ; 
    qgs[0] = MakeInputPhotonGenstep_(input_photon, fr );   
    NP* ipgs = NPX::ArrayFromVec<float,quad6>( qgs, 6, 4) ; 
    return ipgs ; 
}

/**
SEvent::MakeInputPhotonGenstep_
---------------------------------

Now called from SEvt::addFrameGenstep (formerly from SEvt::setFrame)
Note that the only thing taken from the *input_photon* is the 
number of photons so this can work with either local or 
transformed *input_photon*. 

The m2w transform from the frame is copied into the genstep.  
HMM: is that actually used ? Because the frame is also persisted. 

**/

quad6 SEvent::MakeInputPhotonGenstep_(const NP* input_photon, const sframe& fr )
{
    LOG(LEVEL) << " input_photon " << NP::Brief(input_photon) ;  

    quad6 ipgs ; 
    ipgs.zero(); 
    ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON ); 
    ipgs.set_numphoton(  input_photon->shape[0]  ); 
    fr.m2w.write(ipgs); // copy fr.m2w into ipgs.q2,q3,q4,q5 
    return ipgs ; 
}





/**
SEvent::MakeTorchGenstep
--------------------------

Canonically invoked from SEvt::AddTorchGenstep 
which seems to need user code invokation. 
HMM: perhaps SEventConfig to do this in standardized place ?

**/

NP* SEvent::MakeTorchGenstep(int idx){    return MakeGenstep( OpticksGenstep_TORCH, idx ) ; }
NP* SEvent::MakeCerenkovGenstep(int idx){ return MakeGenstep( OpticksGenstep_CERENKOV, idx ) ; }
NP* SEvent::MakeScintGenstep(int idx){    return MakeGenstep( OpticksGenstep_SCINTILLATION, idx ) ; }
NP* SEvent::MakeCarrierGenstep(int idx){  return MakeGenstep( OpticksGenstep_CARRIER, idx ) ; }


/**
SEvent::MakeGenstep
---------------------

For varying photons per event need to get the SEvt index here::

   BP=SEvent::MakeTorchGenstep ~/opticks/CSGOptiX/cxs_min.sh run

**/


NP* SEvent::MakeGenstep( int gentype, int index )
{
    bool with_index = index != -1 ; 
    if(with_index) assert( index > 0 );  // SEvt::index is 1-based 
    int num_ph = with_index ? SEventConfig::NumPhoton(index-1) : ssys::getenvint("SEvent_MakeGenstep_num_ph", 100 ) ; 
    bool dump = ssys::getenvbool("SEvent_MakeGenstep_dump"); 
    unsigned num_gs = 1 ; 

    LOG(info) 
        << " gentype " << gentype
        << " index (1-based) " << index
        << " with_index " << ( with_index ? "YES" : "NO " )
        << " num_ph " << num_ph 
        << " dump " << dump
        ; 

    NP* gs = NP::Make<float>(num_gs, 6, 4 );  
    switch(gentype)
    {
        case  OpticksGenstep_TORCH:         FillGenstep<storch>(   gs, num_ph, dump) ; break ; 
        case  OpticksGenstep_CERENKOV:      FillGenstep<scerenkov>(gs, num_ph, dump) ; break ; 
        case  OpticksGenstep_SCINTILLATION: FillGenstep<sscint>(   gs, num_ph, dump) ; break ; 
        case  OpticksGenstep_CARRIER:       FillGenstep<scarrier>( gs, num_ph, dump) ; break ; 
    }
    return gs ; 
}

template<typename T>
void SEvent::FillGenstep( NP* gs, int numphoton_per_genstep, bool dump )
{
    T* tt = (T*)gs->bytes() ; 
    for(int i=0 ; i < gs->shape[0] ; i++ ) 
    {
        unsigned genstep_id = i ; 
        T::FillGenstep( tt[i], genstep_id, numphoton_per_genstep, dump ) ; 
    }
}

template void SEvent::FillGenstep<storch>(    NP* gs, int numphoton_per_genstep, bool dump ); 
template void SEvent::FillGenstep<scerenkov>( NP* gs, int numphoton_per_genstep, bool dump ); 
template void SEvent::FillGenstep<sscint>(    NP* gs, int numphoton_per_genstep, bool dump ); 
template void SEvent::FillGenstep<scarrier>(  NP* gs, int numphoton_per_genstep, bool dump ); 


/**
SEvent::MakeSeed
------------------

Creates seed array which provides genstep reference indices
for each photon slot. 

Normally this is done on device using involved thrust, 
here is a simple CPU implementation of that.
 
**/

NP* SEvent::MakeSeed( const NP* gs )
{
    assert( gs->has_shape(-1,6,4) );  
    int num_gs = gs->shape[0] ; 
    const storch* tt = (storch*)gs->bytes() ; 
    // storch type is used only to access numphoton in the quad6 (0,3) position
    // which all genstep types place at the same offset 
    // HMM: using quad6 would be clearer

    // vector of numphoton in each genstep
    std::vector<int> gsp(num_gs) ;
    for(int i=0 ; i < num_gs ; i++ ) gsp[i] = tt[i].numphoton ;

    int tot_photons = 0 ; 
    for(int i=0 ; i < num_gs ; i++ ) tot_photons += gsp[i] ; 

    NP* se = NP::Make<int>( tot_photons );  
    int* sev = se->values<int>();  

    // duplicate genstep reference index into the seed array 
    // which is the same length as total photons
    int offset = 0 ; 
    for(int i=0 ; i < num_gs ; i++) 
    {   
        int np = gsp[i] ; 
        for(int p=0 ; p < np ; p++) sev[offset+p] = i ; 
        offset += np ; 
    }   
    return se ; 
}





NP* SEvent::MakeCountGenstep(const char* config, int* total ) // static 
{
    std::vector<int>* photon_counts_per_genstep = nullptr ; 
    if( config == nullptr )
    {
        (*photon_counts_per_genstep) = { 3, 5, 2, 0, 1, 3, 4, 2, 4 }; 
    }
    return MakeCountGenstep(*photon_counts_per_genstep, total);
}

/**
SEvent::MakeCountGenstep
---------------------------

Used by qudarap/tests/QEventTest.cc

**/


NP* SEvent::MakeCountGenstep(const std::vector<int>& counts, int* total ) // static 
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




