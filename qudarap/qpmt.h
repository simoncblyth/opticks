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


#include "s_pmt.h"


template<typename F>
struct qpmt
{
    enum { L0, L1, L2, L3 } ;

    static constexpr const F hc_eVnm = 1239.84198433200208455673  ;
    static constexpr const F zero = 0. ;
    static constexpr const F one = 1. ;
    // constexpr should mean any double conversions happen at compile time ?

    qprop<F>* rindex_prop ;
    qprop<F>* qeshape_prop ;
    qprop<F>* cetheta_prop ;
    qprop<F>* cecosth_prop ;

    F*        thickness ;
    F*        lcqs ;
    int*      i_lcqs ;  // int* "view" of lcqs memory

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
    // loosely follow SPMT.h
    QPMT_METHOD int  get_lpmtcat_from_lpmtid(  int lpmtid  ) const  ;
    QPMT_METHOD int  get_lpmtcat_from_lpmtidx( int lpmtidx ) const  ;
    QPMT_METHOD F    get_qescale_from_lpmtid(  int lpmtid  ) const  ;
    QPMT_METHOD F    get_qescale_from_lpmtidx( int lpmtidx ) const  ;

    QPMT_METHOD F    get_lpmtcat_qe( int pmtcat, F energy_eV ) const ;
    QPMT_METHOD F    get_lpmtcat_ce( int pmtcat, F theta ) const ;

    QPMT_METHOD F    get_lpmtcat_rindex(    int lpmtcat, int layer, int prop, F energy_eV ) const ;
    QPMT_METHOD F    get_lpmtcat_rindex_wl( int lpmtcat, int layer, int prop, F wavelength_nm ) const ;


    QPMT_METHOD void get_lpmtcat_stackspec( F* spec16, int pmtcat, F energy_eV ) const ;

    QPMT_METHOD void get_lpmtid_stackspec(          F* spec16, int lpmtid, F energy_eV ) const ;
    QPMT_METHOD void get_lpmtid_stackspec_ce_acosf( F* spec16, int lpmtid, F energy_eV, F lposcost ) const ;
    QPMT_METHOD void get_lpmtid_stackspec_ce(       F* spec15, int lpmtid, F energy_eV, F lposcost ) const ;


#ifdef WITH_CUSTOM4
    QPMT_METHOD void get_lpmtid_SPEC(   F* spec16 , int lpmtid, F wavelength_nm ) const ;
    QPMT_METHOD void get_lpmtid_SPEC_ce(F* spec16 , int lpmtid, F wavelength_nm, F lposcost ) const ;
    QPMT_METHOD void get_lpmtid_LL(  F* ll_128  , int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ;
    QPMT_METHOD void get_lpmtid_COMP(F* comp_32 , int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ;
    QPMT_METHOD void get_lpmtid_ART( F* art_16  , int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ;
    QPMT_METHOD void get_lpmtid_ARTE(  F* ARTE  , int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm ) const ;
    QPMT_METHOD void get_lpmtid_ATQC(  F* ATQC  , int lpmtid, F wavelength_nm, F minus_cos_theta, F dot_pol_cross_mom_nrm, F lposcost ) const ;
#endif

#endif
};

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)

template<typename F>
inline QPMT_METHOD int qpmt<F>::get_lpmtcat_from_lpmtid( int lpmtid ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);
    return lpmtidx < s_pmt::NUM_CD_LPMT_AND_WP && lpmtidx > -1 ? i_lcqs[lpmtidx*2+0] : -2 ;
}

template<typename F>
inline QPMT_METHOD int qpmt<F>::get_lpmtcat_from_lpmtidx( int lpmtidx ) const
{
    return lpmtidx < s_pmt::NUM_CD_LPMT_AND_WP && lpmtidx > -1 ? i_lcqs[lpmtidx*2+0] : -2 ;
}

template<typename F>
inline QPMT_METHOD F qpmt<F>::get_qescale_from_lpmtid( int lpmtid ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);
    return lpmtidx < s_pmt::NUM_CD_LPMT_AND_WP && lpmtidx > -1 ? lcqs[lpmtidx*2+1] : -2.f ;
}
template<typename F>
inline QPMT_METHOD F qpmt<F>::get_qescale_from_lpmtidx( int lpmtidx ) const
{
    return lpmtidx < s_pmt::NUM_CD_LPMT_AND_WP && lpmtidx > -1 ? lcqs[lpmtidx*2+1] : -2.f ;
}


template<typename F>
inline QPMT_METHOD F qpmt<F>::get_lpmtcat_qe( int lpmtcat, F energy_eV ) const
{
    return lpmtcat > -1 && lpmtcat < s_pmt::NUM_CAT ? qeshape_prop->interpolate( lpmtcat, energy_eV ) : -1.f ;
}

/**
qpmt::get_lpmtcat_ce
---------------------

theta_radians range 0->pi/2


              cos(th)=1
                th=0

                  |
               +  |  +
           +      |      +
                  |
         +        |         +
        +---------+----------+--   th = pi/2
                                cos(th)=0

**/

template<typename F>
inline QPMT_METHOD F qpmt<F>::get_lpmtcat_ce( int lpmtcat, F theta_radians ) const
{
    //return lpmtcat > -1 && lpmtcat < qpmt_NUM_CAT ? cetheta_prop->interpolate( lpmtcat, theta_radians ) : -1.f ;
    return lpmtcat > -1 && lpmtcat < s_pmt::NUM_CAT ? cetheta_prop->interpolate( lpmtcat, theta_radians ) : -1.f ;
}
template<typename F>
inline QPMT_METHOD F qpmt<F>::get_lpmtcat_rindex( int lpmtcat, int layer, int prop, F energy_eV ) const
{
    //const unsigned idx = lpmtcat*qpmt_NUM_LAYR*qpmt_NUM_PROP + layer*qpmt_NUM_PROP + prop ;
    const unsigned idx = lpmtcat*s_pmt::NUM_LAYR*s_pmt::NUM_PROP + layer*s_pmt::NUM_PROP + prop ;
    return rindex_prop->interpolate( idx, energy_eV ) ;
}
template<typename F>
inline QPMT_METHOD F qpmt<F>::get_lpmtcat_rindex_wl( int lpmtcat, int layer, int prop, F wavelength_nm ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;
    return get_lpmtcat_rindex(lpmtcat, layer, prop, energy_eV );
}



template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtcat_stackspec( F* spec, int lpmtcat, F energy_eV ) const
{
    //const unsigned idx = lpmtcat*qpmt_NUM_LAYR*qpmt_NUM_PROP ;
    //const unsigned idx0 = idx + L0*qpmt_NUM_PROP ;
    //const unsigned idx1 = idx + L1*qpmt_NUM_PROP ;
    //const unsigned idx2 = idx + L2*qpmt_NUM_PROP ;

    const unsigned idx = lpmtcat*s_pmt::NUM_LAYR*s_pmt::NUM_PROP ;
    const unsigned idx0 = idx + L0*s_pmt::NUM_PROP ;
    const unsigned idx1 = idx + L1*s_pmt::NUM_PROP ;
    const unsigned idx2 = idx + L2*s_pmt::NUM_PROP ;


    spec[0*4+0] = rindex_prop->interpolate( idx0+0u, energy_eV );
    spec[0*4+1] = zero ;
    spec[0*4+2] = zero ;

    spec[1*4+0] = rindex_prop->interpolate( idx1+0u, energy_eV );
    spec[1*4+1] = rindex_prop->interpolate( idx1+1u, energy_eV );
    //spec[1*4+2] = thickness[lpmtcat*qpmt_NUM_LAYR+L1] ;
    spec[1*4+2] = thickness[lpmtcat*s_pmt::NUM_LAYR+L1] ;

    spec[2*4+0] = rindex_prop->interpolate( idx2+0u, energy_eV );
    spec[2*4+1] = rindex_prop->interpolate( idx2+1u, energy_eV );
    //spec[2*4+2] = thickness[lpmtcat*qpmt_NUM_LAYR+L2] ;
    spec[2*4+2] = thickness[lpmtcat*s_pmt::NUM_LAYR+L2] ;

    spec[3*4+0] = one ;  // Vacuum RINDEX
    spec[3*4+1] = zero ;
    spec[3*4+2] = zero ;

    // "4th" column untouched, as pmtid info goes in there
}



template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec( F* spec, int lpmtid, F energy_eV ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);
    const int& lpmtcat = i_lcqs[lpmtidx*2+0] ;
    // printf("//qpmt::get_lpmtidx_stackspec lpmtid %d lpmtcat %d \n", lpmtid, lpmtcat );

    const F& qe_scale = lcqs[lpmtidx*2+1] ;
    const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ;
    const F qe = qe_scale*qe_shape ;

    spec[0*4+3] = lpmtcat ;
    spec[1*4+3] = qe_scale ;
    spec[2*4+3] = qe_shape ;
    spec[3*4+3] = qe ;

    get_lpmtcat_stackspec( spec, lpmtcat, energy_eV );
}


/**
get_lpmtid_stackspec_ce_acosf (see alternative get_lpmtid_stackspec_ce)
-------------------------------------------------------------------------

This uses cetheta_prop interpolation forcing use of acosf to get theta

lposcost
    local position cosine theta,
    expected range 1->0 (as front of PMT is +Z)
    so theta_radians expected 0->pi/2

Currently called by qpmt::get_lpmtid_ATQC

**/


template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec_ce_acosf( F* spec, int lpmtid, F energy_eV, F lposcost ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);
    const int& lpmtcat = i_lcqs[lpmtidx*2+0] ;

    const F& qe_scale = lcqs[lpmtidx*2+1] ;
    const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ;
    const F qe = qe_scale*qe_shape ;

    const F theta_radians = acosf(lposcost);
    const F ce = cetheta_prop->interpolate( lpmtcat, theta_radians );

    spec[0*4+3] = lpmtcat ;       //  3
    spec[1*4+3] = ce ;            //  7
    spec[2*4+3] = qe_shape ;      // 11
    spec[3*4+3] = qe ;            // 15

    get_lpmtcat_stackspec( spec, lpmtcat, energy_eV );

    //printf("//qpmt::get_lpmtidx_stackspec lpmtid %d lpmtidx %d lpmtcat %d lposcost %7.3f acosf_lposcost %7.3f ce %7.3f spec[7] %7.3f \n", lpmtid, lpmtidx, lpmtcat, lposcost, acosf_lposcost, ce, spec[7] );
}



/**
get_lpmtid_stackspec_ce
-------------------------

This uses cecosth_prop interpolation avoiding use of acosf
as directly interpolate in the cosine.

lposcost
    local position cosine theta,
    expected range 1->0 (as front of PMT is +Z)
    so theta_radians expected 0->pi/2

Potentially called by qpmt::get_lpmtid_ATQC

**/


template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec_ce( F* spec, int lpmtid, F energy_eV, F lposcost ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);
    const int& lpmtcat = i_lcqs[lpmtidx*2+0] ;
    const F& qe_scale = lcqs[lpmtidx*2+1] ;

    const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ;
    const F qe = qe_scale*qe_shape ;

    const F ce = cecosth_prop->interpolate( lpmtcat, lposcost );

    spec[0*4+3] = lpmtcat ;       //  3
    spec[1*4+3] = ce ;            //  7
    spec[2*4+3] = qe_shape ;      // 11
    spec[3*4+3] = qe ;            // 15

    get_lpmtcat_stackspec( spec, lpmtcat, energy_eV );

    //printf("//qpmt::get_lpmtid_stackspec_ce lpmtid %d lpmtidx %d lpmtcat %d lposcost %7.3f  ce %7.3f spec[7] %7.3f \n", lpmtid, lpmtidx, lpmtcat, lposcost, ce, spec[7] );
}







#ifdef WITH_CUSTOM4

/**
qpmt::get_lpmtid_SPEC
-----------------------

Gets the inputs to the TMM stack calculation
customized for particular lpmtid.

1. HMM: has to convert wavelength_nm to energy_eV
   for every photon... could convert to wavelength domain CPU side
   to avoid the need to do that

**/

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_SPEC(
    F* spec_16,
    int lpmtid,
    F wavelength_nm ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;
    get_lpmtid_stackspec( spec_16, lpmtid, energy_eV );
}

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_SPEC_ce(
    F* spec_16,
    int lpmtid,
    F wavelength_nm,
    F lposcost ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;
    get_lpmtid_stackspec_ce_acosf( spec_16, lpmtid, energy_eV, lposcost );
}



/**
qpmt::get_lpmtid_LL
----------------------

Full details of all the TMM layers for debugging.

**/

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_LL(
    F* ll_128,
    int lpmtid,
    F wavelength_nm,
    F minus_cos_theta,
    F dot_pol_cross_mom_nrm ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;

    F spec[16] ;
    get_lpmtid_stackspec( spec, lpmtid, energy_eV );

    Stack<F,4> stack(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, spec, 16u );
    const F* stack_ll = stack.ll[0].cdata() ;

    for(int i=0 ; i < 128 ; i++ ) ll_128[i] = stack_ll[i] ;
}

/**
qpmt::get_lpmtid_COMP
----------------------

Full details of the composite TMM layer for debugging.

**/

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_COMP(
    F* comp_32,
    int lpmtid,
    F wavelength_nm,
    F minus_cos_theta,
    F dot_pol_cross_mom_nrm ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;

    F spec[16] ;
    get_lpmtid_stackspec( spec, lpmtid, energy_eV );

    Stack<F,4> stack(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, spec, 16u );
    const F* stack_comp = stack.comp.cdata() ;
    for(int i=0 ; i < 32 ; i++ ) comp_32[i] = stack_comp[i] ;
}

/**
qpmt::get_lpmtid_ART
----------------------

4x4 ART matrix with A,R,T for S/P/av/actual polarizations

**/


template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_ART(
    F* art_16,
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

    for(int i=0 ; i < 16 ; i++ ) art_16[i] = stack_art[i] ;
}

/**
qpmt::get_lpmtid_ARTE
-----------------------

lpmtid and polarization customized TMM calc of::

   A : theAbsorption
   R : theReflectivity
   T : theTransmittance
   E : theEfficiency (more specifically QE)

Q: Does theReflectivity+theTransmittace = 1.f ?
A: YES, by construction because A+R+T = one (triplet prob)

   so : R+T=one-A

   R+T
   ----- = one
   one-A


**/

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_ARTE(
    F* ARTE,
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

#ifdef MOCK_CURAND_DEBUG
    printf("//qpmt::get_lpmtid_ARTE lpmtid %d energy_eV %7.3f _qe %7.3f \n", lpmtid, energy_eV, _qe );
#endif


    Stack<F,4> stack ;

    if( minus_cos_theta < zero )
    {
        stack.calc(wavelength_nm, -one, zero, ss, 16u );
        ARTE[3] = _qe/stack.art.A ;

#ifdef MOCK_CURAND_DEBUG
        printf("//qpmt::get_lpmtid_ARTE stack.art.A %7.3f _qe/stack.art.A %7.3f \n", stack.art.A, ARTE[3] );
#endif
    }
    else
    {
        ARTE[3] = zero ;
    }

    stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );

    const F& A = stack.art.A ;
    const F& R = stack.art.R ;
    const F& T = stack.art.T ;

    ARTE[0] = A ;         // aka theAbsorption
    ARTE[1] = R/(one-A) ; // aka theReflectivity
    ARTE[2] = T/(one-A) ; // aka theTransmittance

#ifdef MOCK_CURAND_DEBUG
    std::cout << "//qpmt::get_lpmtid_ARTE stack.calc.2 " << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ARTE wavelength_nm " << wavelength_nm << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ARTE minus_cos_theta " << minus_cos_theta << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ARTE dot_pol_cross_mom_nrm " << dot_pol_cross_mom_nrm << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ARTE " << stack << std::endl ;
#endif

}


/**
qpmt::get_lpmtid_ATQC
------------------------

Invoked from qsim::propagate_at_surface_CustomART


ATQC[0] A
   absorption

ATQC[1] T
   transmission, scaled by 1/(1-A) to make R+T = 1

ATQC[2] Q
   QE quantum efficiency, depending on photon energy
   and PMT TMM parameters

ATQC[3] C
   CE collection efficiency, depending on theta of local position on PMT surface
   obtained by interpolation over theta domain (OR maybe in future costheta domain)


TODO: compare between the alternates::

    get_lpmtid_stackspec_ce_acosf   // interpolates ce in theta of local position in PMT frame
    get_lpmtid_stackspec_ce         // interpolates ce in costheta of local position in PMT frame

**/

template<typename F>
inline QPMT_METHOD void qpmt<F>::get_lpmtid_ATQC(
    F* ATQC,
    int lpmtid,
    F wavelength_nm,
    F minus_cos_theta,
    F dot_pol_cross_mom_nrm,
    F lposcost ) const
{
    const F energy_eV = hc_eVnm/wavelength_nm ;

    F spec[16] ;

    get_lpmtid_stackspec_ce_acosf( spec, lpmtid, energy_eV, lposcost );
  //get_lpmtid_stackspec_ce(       spec, lpmtid, energy_eV, lposcost );


    const F* ss = spec ;
    const F& ce = spec[7] ;
    ATQC[3] = ce ;

    const F& _qe = spec[15] ;

#ifdef MOCK_CURAND_DEBUG
    printf("//qpmt::get_lpmtid_ARTQC lpmtid %d energy_eV %7.3f _qe %7.3f lposcost %7.3f ce %7.3f  \n", lpmtid, energy_eV, _qe, lposcost, ce );
#endif

    Stack<F,4> stack ;

    if( minus_cos_theta < zero )
    {
        stack.calc(wavelength_nm, -one, zero, ss, 16u );
        ATQC[2] = _qe/stack.art.A ;

#ifdef MOCK_CURAND_DEBUG
        printf("//qpmt::get_lpmtid_ATQC stack.art.A %7.3f _qe/stack.art.A=ATQC[2] %7.3f  ce=ATQC[3] %7.3f  \n", stack.art.A, ATQC[2], ATQC[3] );
#endif
    }
    else
    {
        ATQC[2] = zero ;
    }

    stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );

    const F& A = stack.art.A ;
    const F& T = stack.art.T ;

    ATQC[0] = A ;         // aka theAbsorption
    ATQC[1] = T/(one-A) ; // aka theTransmittance


#ifdef MOCK_CURAND_DEBUG
    std::cout << "//qpmt::get_lpmtid_ATQC stack.calc.2 " << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ATQC wavelength_nm " << wavelength_nm << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ATQC minus_cos_theta " << minus_cos_theta << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ATQC dot_pol_cross_mom_nrm " << dot_pol_cross_mom_nrm << std::endl ;
    std::cout << "//qpmt::get_lpmtid_ATQC " << stack << std::endl ;
#endif

}





// end WITH_CUSTOM4
#endif

// end CUDA/MOCK_CUDA
#endif


#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
#else
template struct QUDARAP_API qpmt<float>;
//template struct QUDARAP_API qpmt<double>;
#endif



