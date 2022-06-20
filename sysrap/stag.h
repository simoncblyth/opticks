#pragma once
/**
stag.h : random consumption tags for simulation alignment purposes
=====================================================================


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define STAG_METHOD __device__ __forceinline__
#else
#    define STAG_METHOD inline 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iomanip>
#endif


enum {
   stag_undef     =  0,
   stag_to_sc     =  1, 
   stag_to_ab     =  2,
   stag_to_re     =  3,
   stag_re_wl     =  4,
   stag_re_mom_ph =  5,
   stag_re_mom_ct =  6,
   stag_re_pol_ph =  7,
   stag_re_pol_ct =  8,
   stag_at_bo     =  9,
   stag_at_rf     = 10,
   stag_sf_sd     = 11,
   stag_sf_bu     = 12,
   stag_hp_ph     = 13,
   stag_hp_ct     = 14,
   stag_sc_u0     = 15,
   stag_sc_u1     = 16,
   stag_sc_u2     = 17,
   stag_sc_u3     = 18,
   stag_sc_u4     = 19
}; 

struct stag
{
    static constexpr const unsigned NSEQ = 2 ;  
    static constexpr const unsigned BITS = 5 ;   // (0x1 << 5)-1 = 31  : up to 32 enumerations in 5 bits per slot   
    static constexpr const unsigned long long MASK = ( 0x1ull << BITS ) - 1ull ;   
    static constexpr const unsigned SLOTMAX = 64/BITS ;  // 64//5 = 12 so can fit 12 tags into each seqtag 64 bits, with NSEQ 2 that allows 24 tags  

    // int slot = 0 ;  having the slot inside the struct is inconvenient for purpose of persisting into a stag array eg: evt->tag[idx] = tag 
    // HMM could operate slotless by slotting into first with no bits set ?
    unsigned long long seqtag[NSEQ] ;
 
    STAG_METHOD void zero(); 
    STAG_METHOD void add(unsigned& slot, unsigned tag, float u=0.f );   // passing the random only needed during dev to check are tagging all consumption 
    STAG_METHOD unsigned get(unsigned slot_) const ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static constexpr const char* undef_ = "_" ; 
    static constexpr const char* undef_note = "undef" ; 

    static constexpr const char* to_sc_ = "to_sc" ; 
    static constexpr const char* to_sc_note = "qsim::propagate_to_boundary u_scattering" ; 

    static constexpr const char* to_ab_ = "to_ab" ; 
    static constexpr const char* to_ab_note = "qsim::propagate_to_boundary u_absorption" ; 

    static constexpr const char* to_re_ = "to_re" ; 
    static constexpr const char* to_re_note = "qsim::propagate_to_boundary u_reemit" ; 

    static constexpr const char* re_wl_ = "re_wl" ; 
    static constexpr const char* re_wl_note = "qsim::propagate_to_boundary u_wavelength " ; 

    static constexpr const char* re_mom_ph_ = "re_mom_ph" ; 
    static constexpr const char* re_mom_ph_note = "qsim::propagate_to_boundary re mom uniform_sphere ph " ; 

    static constexpr const char* re_mom_ct_ = "re_mom_ct" ; 
    static constexpr const char* re_mom_ct_note = "qsim::propagate_to_boundary re mom uniform_sphere ct " ;

    static constexpr const char* re_pol_ph_ = "re_pol_ph" ; 
    static constexpr const char* re_pol_ph_note = "qsim::propagate_to_boundary re pol uniform_sphere ph " ; 

    static constexpr const char* re_pol_ct_ = "re_pol_ct" ; 
    static constexpr const char* re_pol_ct_note = "qsim::propagate_to_boundary re pol uniform_sphere ct " ;

    static constexpr const char* at_bo_ = "at_bo" ; 
    static constexpr const char* at_bo_note = "boundary burn" ; 

    static constexpr const char* at_rf_ = "at_rf" ; 
    static constexpr const char* at_rf_note = "u_reflect > TransCoeff" ; 

    static constexpr const char* sf_sd_ = "sf_sd" ; 
    static constexpr const char* sf_sd_note = "qsim::propagate_at_surface ab/sd" ; 

    static constexpr const char* sf_bu_ = "sf_bu" ; 
    static constexpr const char* sf_bu_note = "qsim::propagate_at_surface burn" ; 

    static constexpr const char* hp_ph_ = "hp_ph" ; 
    static constexpr const char* hp_ph_note = "qsim::hemisphere_polarized u_hemipol_phi" ; 

    static constexpr const char* hp_ct_ = "hp_ct" ; 
    static constexpr const char* hp_ct_note = "qsim::hemisphere_polarized cosTheta" ; 

    static constexpr const char* sc_u0_ = "sc_u0" ; 
    static constexpr const char* sc_u0_note = "qsim::rayleigh_scatter u0" ;

    static constexpr const char* sc_u1_ = "sc_u1" ; 
    static constexpr const char* sc_u1_note = "qsim::rayleigh_scatter u1" ;

    static constexpr const char* sc_u2_ = "sc_u2" ; 
    static constexpr const char* sc_u2_note = "qsim::rayleigh_scatter u2" ;

    static constexpr const char* sc_u3_ = "sc_u3" ; 
    static constexpr const char* sc_u3_note = "qsim::rayleigh_scatter u3" ;

    static constexpr const char* sc_u4_ = "sc_u4" ; 
    static constexpr const char* sc_u4_note = "qsim::rayleigh_scatter u4" ;


    static const char* Name(unsigned tag);     
    static const char* Note(unsigned tag);     

    std::string desc() const ; 
    static std::string Desc() ; 
#endif
};



STAG_METHOD void stag::zero()
{
    for(unsigned i=0 ; i < NSEQ ; i++) seqtag[i] = 0ull ; 
}

STAG_METHOD void stag::add(unsigned& slot, unsigned tag, float u)
{
    unsigned iseq = slot/SLOTMAX ; 
    if(iseq < NSEQ) seqtag[iseq] |=  (( tag & MASK ) << BITS*(slot - iseq*SLOTMAX) );
    slot += 1 ; 
}

STAG_METHOD unsigned stag::get(unsigned slot_) const 
{
    unsigned iseq = slot_/SLOTMAX ; 
    return iseq < NSEQ ? ( seqtag[iseq] >> BITS*(slot_ - iseq*SLOTMAX) ) & MASK : 0  ; 
}




#if defined(__CUDACC__) || defined(__CUDABE__)
#else
STAG_METHOD const char* stag::Name(unsigned tag)
{
    const char* s = nullptr ; 
    switch(tag)
    {
        case stag_undef: s = undef_ ; break ; 
        case stag_to_sc: s = to_sc_ ; break ; 
        case stag_to_ab: s = to_ab_ ; break ; 
        case stag_to_re: s = to_re_ ; break ; 
        case stag_re_wl: s = re_wl_ ; break ; 
        case stag_re_mom_ph: s = re_mom_ph_ ; break ; 
        case stag_re_mom_ct: s = re_mom_ct_ ; break ; 
        case stag_re_pol_ph: s = re_pol_ph_ ; break ; 
        case stag_re_pol_ct: s = re_pol_ct_ ; break ; 
        case stag_at_bo: s = at_bo_ ; break ; 
        case stag_at_rf: s = at_rf_ ; break ; 
        case stag_sf_sd: s = sf_sd_ ; break ; 
        case stag_sf_bu: s = sf_bu_ ; break ; 
        case stag_hp_ph: s = hp_ph_ ; break ; 
        case stag_hp_ct: s = hp_ct_ ; break ; 
        case stag_sc_u0: s = sc_u0_ ; break ; 
        case stag_sc_u1: s = sc_u1_ ; break ; 
        case stag_sc_u2: s = sc_u2_ ; break ; 
        case stag_sc_u3: s = sc_u3_ ; break ; 
        case stag_sc_u4: s = sc_u4_ ; break ; 
    }
    return s ; 
}
STAG_METHOD const char* stag::Note(unsigned tag)
{
    const char* s = nullptr ; 
    switch(tag)
    {
        case stag_undef: s = undef_note ; break ; 
        case stag_to_sc: s = to_sc_note ; break ; 
        case stag_to_ab: s = to_ab_note ; break ; 
        case stag_to_re: s = to_re_note ; break ; 
        case stag_re_wl: s = re_wl_note ; break ; 
        case stag_re_mom_ph: s = re_mom_ph_note ; break ; 
        case stag_re_mom_ct: s = re_mom_ct_note ; break ; 
        case stag_re_pol_ph: s = re_pol_ph_note ; break ; 
        case stag_re_pol_ct: s = re_pol_ct_note ; break ; 
        case stag_at_bo: s = at_bo_note ; break ; 
        case stag_at_rf: s = at_rf_note ; break ; 
        case stag_sf_sd: s = sf_sd_note ; break ; 
        case stag_sf_bu: s = sf_bu_note ; break ; 
        case stag_hp_ph: s = hp_ph_note ; break ; 
        case stag_hp_ct: s = hp_ct_note ; break ; 
        case stag_sc_u0: s = sc_u0_note ; break ; 
        case stag_sc_u1: s = sc_u1_note ; break ; 
        case stag_sc_u2: s = sc_u2_note ; break ; 
        case stag_sc_u3: s = sc_u3_note ; break ; 
        case stag_sc_u4: s = sc_u4_note ; break ; 
    }
    return s ; 
}



STAG_METHOD std::string stag::desc() const 
{
    std::stringstream ss ;
    ss << std::hex << std::setw(16) << seqtag << std::dec << " : " ; 
    ss << std::endl ; 
    for(unsigned i=0 ; i < SLOTMAX*NSEQ ; i++) 
    {
        unsigned tag = get(i); 
        const char* name = Name(tag) ; 
        const char* note = Note(tag) ; 
        ss 
             << " i " << std::setw(2) << i 
             << " tag " << std::setw(2) << tag
             << " name " << std::setw(15) << ( name ? name : "-" )  
             << " note " << ( note ? note : "-" ) 
             << std::endl
             ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

STAG_METHOD std::string stag::Desc() 
{
    std::stringstream ss ;

    ss << "stag::Desc " 
       << " BITS " << BITS 
       << " MASK " << MASK
       << " MASK 0x" << std::hex << MASK << std::dec
       << " MASK 0b" << std::bitset<64>(MASK) 
       << " 64/BITS " << 64/BITS
       << " SLOTMAX " << SLOTMAX
       << std::endl 
       ;  

    for(unsigned i=0 ; i <= MASK ; i++) 
    {
        unsigned tag = i ; 
        const char* name = Name(tag) ; 
        const char* note = Note(tag) ; 
        ss 
             << " i " << std::setw(2) << i 
             << " tag " << std::setw(2) << tag
             << " name " << std::setw(15) << ( name ? name : "-" )  
             << " note " << ( note ? note : "-" ) 
             << std::endl
             ; 
    }
    std::string s = ss.str(); 
    return s ; 

}
#endif


