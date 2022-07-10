#pragma once
/**
stag.h : random consumption tags for simulation alignment purposes
=====================================================================

* advantageous for the most common tags to have smaller enum values for ease of presentation/debugging  
* going beyond 32 enum values will force increasing BITS from 5 

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define STAG_METHOD __device__ __forceinline__
#else
#    define STAG_METHOD inline 
#endif

enum {
   stag_undef      =    0,
   stag_to_sci     =    1,  
   stag_to_bnd     =    2, 
   stag_to_sca     =    3, 
   stag_to_abs     =    4,
   stag_at_burn_sf_sd = 5,
   stag_at_ref     =    6,
   stag_sf_burn    =    7,
   stag_sc         =    8,
   stag_to_ree     =    9,
   stag_re_wl      =   10,
   stag_re_mom_ph  =   11,
   stag_re_mom_ct  =   12,
   stag_re_pol_ph  =   13,
   stag_re_pol_ct  =   14,
   stag_hp_ph      =   15
//   stag_hp_ct      =   16
};    // HMM: squeezing to 0:15 allows reducing stag::BITS from 5 to 4 which packs nicer

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <string>
#include <sstream>
#include <iomanip>
#include <bitset>

struct stagc
{
    static const char* Name(unsigned tag);     
    static const char* Note(unsigned tag);     
    static std::string Desc() ; 

    static constexpr const char* undef_ = "_" ;    // 0 
    static constexpr const char* undef_note = "undef" ; 

    static constexpr const char* to_sci_ = "to_sci" ;   // 1
    static constexpr const char* to_sci_note = "qsim::propagate_to_boundary u_to_sci burn" ; 

    static constexpr const char* to_bnd_ = "to_bnd" ;   // 2 
    static constexpr const char* to_bnd_note = "qsim::propagate_to_boundary u_to_bnd burn" ; 

    static constexpr const char* to_sca_ = "to_sca" ;   // 3 
    static constexpr const char* to_sca_note = "qsim::propagate_to_boundary u_scattering" ; 

    static constexpr const char* to_abs_ = "to_abs" ;   // 4 
    static constexpr const char* to_abs_note = "qsim::propagate_to_boundary u_absorption" ; 

    static constexpr const char* at_burn_sf_sd_ = "at_burn_sf_sd" ;  // 5
    static constexpr const char* at_burn_sf_sd_note = "at_boundary_burn at_surface ab/sd " ; 

    static constexpr const char* at_ref_ = "at_ref" ;    // 6 
    static constexpr const char* at_ref_note = "u_reflect > TransCoeff" ; 

    static constexpr const char* sf_burn_ = "sf_burn" ;  // 7 
    static constexpr const char* sf_burn_note = "qsim::propagate_at_surface burn" ; 

    static constexpr const char* sc_ = "sc" ; // 8
    static constexpr const char* sc_note = "qsim::rayleigh_scatter" ;

    static constexpr const char* to_ree_ = "to_ree" ;   // 9 
    static constexpr const char* to_ree_note = "qsim::propagate_to_boundary u_reemit" ; 

    static constexpr const char* re_wl_ = "re_wl" ;     // 10
    static constexpr const char* re_wl_note = "qsim::propagate_to_boundary u_wavelength " ; 

    static constexpr const char* re_mom_ph_ = "re_mom_ph" ;  // 11
    static constexpr const char* re_mom_ph_note = "qsim::propagate_to_boundary re mom uniform_sphere ph " ; 

    static constexpr const char* re_mom_ct_ = "re_mom_ct" ;  // 12
    static constexpr const char* re_mom_ct_note = "qsim::propagate_to_boundary re mom uniform_sphere ct " ;

    static constexpr const char* re_pol_ph_ = "re_pol_ph" ;  // 13
    static constexpr const char* re_pol_ph_note = "qsim::propagate_to_boundary re pol uniform_sphere ph " ; 

    static constexpr const char* re_pol_ct_ = "re_pol_ct" ;  // 14
    static constexpr const char* re_pol_ct_note = "qsim::propagate_to_boundary re pol uniform_sphere ct " ;

    static constexpr const char* hp_ph_ = "hp_ph" ;  // 15
    static constexpr const char* hp_ph_note = "qsim::hemisphere_polarized u_hemipol_phi" ; 

    //static constexpr const char* hp_ct_ = "hp_ct" ;  // 16 
    //static constexpr const char* hp_ct_note = "qsim::hemisphere_polarized cosTheta" ; 



};

STAG_METHOD const char* stagc::Name(unsigned tag)
{
    const char* s = nullptr ; 
    switch(tag)
    {
        case stag_undef:     s = undef_     ; break ;  // 0
        case stag_to_sci:    s = to_sci_    ; break ;  // 1  
        case stag_to_bnd:    s = to_bnd_    ; break ;  // 2 
        case stag_to_sca:    s = to_sca_    ; break ;  // 3 
        case stag_to_abs:    s = to_abs_    ; break ;  // 4 
        case stag_at_burn_sf_sd:   s = at_burn_sf_sd_   ; break ;  // 5 
        case stag_at_ref:    s = at_ref_    ; break ;  // 6 
        case stag_sf_burn:   s = sf_burn_   ; break ;  // 7
        case stag_sc:        s = sc_        ; break ;  // 8
        case stag_to_ree:    s = to_ree_    ; break ;  // 9
        case stag_re_wl:     s = re_wl_     ; break ;  // 10
        case stag_re_mom_ph: s = re_mom_ph_ ; break ;  // 11
        case stag_re_mom_ct: s = re_mom_ct_ ; break ;  // 12 
        case stag_re_pol_ph: s = re_pol_ph_ ; break ;  // 13
        case stag_re_pol_ct: s = re_pol_ct_ ; break ;  // 14
        case stag_hp_ph:     s = hp_ph_     ; break ;  // 15 
       //case stag_hp_ct:     s = hp_ct_     ; break ;  // 16
    }
    return s ; 
}
STAG_METHOD const char* stagc::Note(unsigned tag)
{
    const char* s = nullptr ; 
    switch(tag)
    {
        case stag_undef:           s = undef_note ; break ;   // 0
        case stag_to_sci:          s = to_sci_note ; break ;  // 1 
        case stag_to_bnd:          s = to_bnd_note ; break ;  // 2
        case stag_to_sca:          s = to_sca_note ; break ;  // 3 
        case stag_to_abs:          s = to_abs_note ; break ;  // 4
        case stag_at_burn_sf_sd:   s = at_burn_sf_sd_note ; break ; // 5 
        case stag_at_ref:          s = at_ref_note ; break ;  // 6
        case stag_sf_burn:         s = sf_burn_note ; break ;  // 7
        case stag_sc:              s = sc_note      ; break ;  // 8
        case stag_to_ree:          s = to_ree_note ; break ;   // 9
        case stag_re_wl:           s = re_wl_note ; break ;    // 10
        case stag_re_mom_ph:       s = re_mom_ph_note ; break ; // 11
        case stag_re_mom_ct:       s = re_mom_ct_note ; break ;  // 12
        case stag_re_pol_ph:       s = re_pol_ph_note ; break ;  // 13
        case stag_re_pol_ct:       s = re_pol_ct_note ; break ;  // 14
        case stag_hp_ph:           s = hp_ph_note ; break ;     // 15
        //case stag_hp_ct:           s = hp_ct_note ; break ;     // 16 
    }
    return s ; 
}

STAG_METHOD std::string stagc::Desc() 
{
    std::stringstream ss ;
    for(unsigned i=0 ; i <= 31 ; i++) 
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

struct stag
{
    static constexpr const unsigned NSEQ = 4 ;   // NB MUST MATCH stag.py:NSEQ
    static constexpr const unsigned BITS = 4 ;   // (0x1 << 5)-1 = 31  : up to 32 enumerations in 5 bits per slot   
    static constexpr const unsigned long long MASK = ( 0x1ull << BITS ) - 1ull ;   
    static constexpr const unsigned SLOTMAX = 64/BITS ;     // eg 64//5 = 12 so can fit 12 tags into each seqtag 64 bits
    static constexpr const unsigned SLOTS = SLOTMAX*NSEQ ;  // eg 24 for BITS = 5 with NSEQ = 2 

    unsigned long long seqtag[NSEQ] ;
 
    STAG_METHOD void zero(); 
    STAG_METHOD void set(unsigned slot, unsigned tag );
    STAG_METHOD unsigned get(unsigned slot_) const ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static std::string Desc() ; 
    std::string desc(unsigned slot) const ; 
    std::string desc() const ; 
#endif

};

struct sflat
{
    static constexpr const unsigned SLOTS = stag::SLOTS ; 
    float flat[SLOTS] ; 
}; 


struct stagr
{
    static constexpr const unsigned SLOTS = stag::SLOTS ; 

    unsigned slot = 0 ; 
    stag  tag = {} ;  
    sflat flat = {} ; 

    STAG_METHOD void add(unsigned tag_, float flat_ );  

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    STAG_METHOD void zero();  
    STAG_METHOD std::string desc(unsigned slot_) const ; 
    STAG_METHOD std::string desc() const ; 
#endif
};

STAG_METHOD void stagr::add(unsigned tag_, float flat_)
{
    //printf("//stagr::add slot %2d tag %2d flat %10.4f SLOTS %d \n", slot, tag_, flat_, SLOTS ); 
    if(slot < SLOTS)
    {
        tag.set(slot, tag_); 
        flat.flat[slot] = flat_ ; 
    }
    slot += 1 ; 
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
STAG_METHOD void stagr::zero(){ *this = {} ; }
STAG_METHOD std::string stagr::desc(unsigned slot_) const 
{
    std::stringstream ss ;
    ss << std::setw(10) << std::fixed << std::setprecision(5) << flat.flat[slot_] << " : " << tag.desc(slot_) ; 
    std::string s = ss.str(); 
    return s ; 
}
STAG_METHOD std::string stagr::desc() const 
{
    std::stringstream ss ;
    ss << "stagr::desc " << std::endl ; 
    for(unsigned i=0 ; i < SLOTS ; i++) ss << desc(i) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}
#endif



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
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
    std::string s = ss.str(); 
    return s ; 
}

STAG_METHOD std::string stag::desc(unsigned slot) const 
{
    unsigned tag = get(slot); 
    const char* name = stagc::Name(tag) ; 
    const char* note = stagc::Note(tag) ; 
    std::stringstream ss ;
    ss 
         << " slot " << std::setw(2) << slot
         << " tag " << std::setw(2) << tag
         << " name " << std::setw(15) << ( name ? name : "-" )  
         << " note " << ( note ? note : "-" ) 
         ; 
    std::string s = ss.str(); 
    return s ; 
}

STAG_METHOD std::string stag::desc() const 
{
    std::stringstream ss ;
    ss << "stag::desc " << std::endl ; 
    for(unsigned i=0 ; i < NSEQ ; i++) ss << std::hex << std::setw(16) << seqtag[i] << std::dec << " : " << std::endl ; 
    for(unsigned i=0 ; i < SLOTS ; i++) ss << desc(i) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}
#endif


STAG_METHOD void stag::zero()
{
    for(unsigned i=0 ; i < NSEQ ; i++) seqtag[i] = 0ull ; 
}
STAG_METHOD void stag::set(unsigned slot, unsigned tag) 
{
    unsigned iseq = slot/SLOTMAX ; 
    if(iseq < NSEQ) seqtag[iseq] |=  (( tag & MASK ) << BITS*(slot - iseq*SLOTMAX) );
}
STAG_METHOD unsigned stag::get(unsigned slot) const 
{
    unsigned iseq = slot/SLOTMAX ; 
    return iseq < NSEQ ? ( seqtag[iseq] >> BITS*(slot - iseq*SLOTMAX) ) & MASK : 0  ; 
}


