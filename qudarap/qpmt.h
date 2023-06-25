#pragma once
/**
qpmt.h
=======


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPMT_METHOD __device__
#else
   #define QPMT_METHOD 
#endif 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "QUDARAP_API_EXPORT.hh"
#endif


template <typename T> struct qprop ;

#include "scuda.h"
#include "squad.h"
#include "qprop.h"

#ifdef WITH_CUSTOM4
#include "C4MultiLayrStack.h"
#endif

template<typename F>
struct qpmt
{
    enum { NUM_CAT = 3, NUM_LAYR = 4, NUM_PROP = 2, NUM_LPMT = 17612 } ;  
    enum { L0, L1, L2, L3 } ; 
    enum { RINDEX, KINDEX, QESHAPE, LPMTCAT_STACKSPEC, LPMTID_STACKSPEC, LPMTID_ART, LPMTID_ARTE } ; 

    static constexpr const F hc_eVnm = 1239.84198433200208455673  ;
    static constexpr const F zero = 0. ; 
    static constexpr const F one = 1. ;   
    // constexpr should mean any double conversions happen at compile time ?

    qprop<F>* rindex_prop ;
    qprop<F>* qeshape_prop ;

    F*        thickness ; 
    F*        lcqs ; 
    int*      i_lcqs ;  // int* "view" of lcqs memory

#if defined(__CUDACC__) || defined(__CUDABE__)
    // loosely follow SPMT.h 
    QPMT_METHOD int  get_lpmtcat(  int pmtid ) const  ; 
    QPMT_METHOD F    get_qescale( int pmtid ) const  ; 
    QPMT_METHOD F    get_lpmtcat_qe( int pmtcat, F energy_eV ) const ; 

    QPMT_METHOD void get_lpmtcat_stackspec( F* spec16, int pmtcat, F energy_eV ) const ; 
    QPMT_METHOD void get_lpmtid_stackspec(  F* spec16, int pmtid,  F energy_eV ) const ; 

#ifdef WITH_CUSTOM4
    QPMT_METHOD void get_lpmtid_ART( F* art16, int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ; 
    QPMT_METHOD void get_lpmtid_ARTE(F* arte4, int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ; 
#endif

#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)

template<typename F>
inline QPMT_METHOD int qpmt<F>::get_lpmtcat( int pmtid ) const 
{
    return pmtid < NUM_LPMT && pmtid > -1 ? i_lcqs[pmtid*2+0] : -2 ; 
}
template<typename F>
inline QPMT_METHOD F qpmt<F>::get_qescale( int pmtid ) const 
{
    return pmtid < NUM_LPMT && pmtid > -1 ? lcqs[pmtid*2+1] : -2.f ; 
}
template<typename F>
inline QPMT_METHOD F qpmt<F>::get_lpmtcat_qe( int lpmtcat, F energy_eV ) const 
{
    return lpmtcat > -1 && lpmtcat < NUM_CAT ? qeshape_prop->interpolate( lpmtcat, energy_eV ) : -1.f ; 
}

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtcat_stackspec( F* spec, int lpmtcat, F energy_eV ) const 
{
    const unsigned idx = lpmtcat*NUM_LAYR*NUM_PROP ; 
    const unsigned idx0 = idx + L0*NUM_PROP ; 
    const unsigned idx1 = idx + L1*NUM_PROP ; 
    const unsigned idx2 = idx + L2*NUM_PROP ; 

    spec[0*4+0] = rindex_prop->interpolate( idx0+0u, energy_eV ); 
    spec[0*4+1] = zero ; 
    spec[0*4+2] = zero ; 

    spec[1*4+0] = rindex_prop->interpolate( idx1+0u, energy_eV ); 
    spec[1*4+1] = rindex_prop->interpolate( idx1+1u, energy_eV ); 
    spec[1*4+2] = thickness[lpmtcat*NUM_LAYR+L1] ;

    spec[2*4+0] = rindex_prop->interpolate( idx2+0u, energy_eV ); 
    spec[2*4+1] = rindex_prop->interpolate( idx2+1u, energy_eV ); 
    spec[2*4+2] = thickness[lpmtcat*NUM_LAYR+L2] ;

    spec[3*4+0] = one ;  // Vacuum RINDEX
    spec[3*4+1] = zero ; 
    spec[3*4+2] = zero ; 

    // "4th" column untouched, as pmtid info goes in there 
}


template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec( F* spec, int lpmtid, F energy_eV ) const 
{
    const int& lpmtcat = i_lcqs[lpmtid*2+0] ; 
    // printf("//qpmt::get_lpmtid_stackspec lpmtid %d lpmtcat %d \n", lpmtid, lpmtcat );  

    const F& qe_scale = lcqs[lpmtid*2+1] ; 
    const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ; 
    const F qe = qe_scale*qe_shape ; 

    spec[0*4+3] = lpmtcat ; 
    spec[1*4+3] = qe_scale ; 
    spec[2*4+3] = qe_shape ; 
    spec[3*4+3] = qe ; 

    get_lpmtcat_stackspec( spec, lpmtcat, energy_eV ); 
}

#ifdef WITH_CUSTOM4


template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_ART(
    F* art16,   
    int lpmtid, 
    F wavelength_nm, 
    F minus_cos_theta, 
    F dot_pol_cross_mom_nrm ) const 
{
    const F energy_eV = hc_eVnm/wavelength_nm ; 

    F spec[16] ; 
    get_lpmtid_stackspec( spec, lpmtid, energy_eV ); 

    Stack<F,4> stack(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, spec, 16u );
    const F* stack_art = stack.art.cdata() ; 

    for(int i=0 ; i < 16 ; i++ ) art16[i] = stack_art[i] ; 
}



template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_ARTE(
    F* arte4,   
    int lpmtid, 
    F wavelength_nm, 
    F minus_cos_theta, 
    F dot_pol_cross_mom_nrm ) const 
{
    const F energy_eV = hc_eVnm/wavelength_nm ; 

    F spec[16] ; 
    get_lpmtid_stackspec( spec, lpmtid, energy_eV ); 

    const F* ss = spec ;
    const F& _qe = spec[15] ;

    Stack<F,4> stack ; 

    if( minus_cos_theta < zero )
    {
        stack.calc(wavelength_nm, -one, zero, ss, 16u );
        arte4[3] = _qe/stack.art.A ; 
    }
    else
    {
        arte4[3] = zero ;  
    }

    stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );
   
    const F& A = stack.art.A ; 
    const F& R = stack.art.R ; 
    const F& T = stack.art.T ; 
 
    arte4[0] = A ;         // aka theAbsorption
    arte4[1] = R/(one-A) ; // aka theReflectivity
    arte4[2] = T/(one-A) ; // aka theTransmittance
}

#endif

#endif



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template struct QUDARAP_API qpmt<float>;
//template struct QUDARAP_API qpmt<double>;
#endif



