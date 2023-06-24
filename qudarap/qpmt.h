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

template<typename T>
struct qpmt
{
    enum { NUM_CAT = 3, NUM_LAYR = 4, NUM_PROP = 2, NUM_LPMT = 17612 } ;  
    enum { L0, L1, L2, L3 } ; 
    enum { RINDEX, KINDEX, QESHAPE, LPMTCAT_STACKSPEC, LPMTID_STACKSPEC } ; 

    static constexpr const T hc_eVnm = 1239.84198433200208455673  ;
    static constexpr const T zero = 0. ; 

    qprop<T>* rindex_prop ;
    qprop<T>* qeshape_prop ;

    T*        thickness ; 
    T*        lcqs ; 
    int*      i_lcqs ;  // int* "view" of lcqs memory

#if defined(__CUDACC__) || defined(__CUDABE__)
    // follow SPMT.h API 
    QPMT_METHOD int  get_lpmtcat(  int pmtid ) const  ; 
    QPMT_METHOD T    get_qescale( int pmtid ) const  ; 
    QPMT_METHOD T    get_lpmtcat_qe( int pmtcat, T energy_eV ) const ; 

    QPMT_METHOD void get_lpmtcat_stackspec( quad4& spec, int pmtcat, T energy_eV ) const ; 
    QPMT_METHOD void get_lpmtid_stackspec(  quad4& spec, int pmtid,  T energy_eV ) const ; 

#ifdef WITH_CUSTOM4
    QPMT_METHOD void get_lpmtid_ARTE(float4& ARTE, int lpmtid, T wavelength_nm, T minus_cos_theta, T dot_pol_cross_mom_nrm ) const ; 
#endif

#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)

template<typename T>
inline QPMT_METHOD int qpmt<T>::get_lpmtcat( int pmtid ) const 
{
    return pmtid < NUM_LPMT && pmtid > -1 ? i_lcqs[pmtid*2+0] : -2 ; 
}
template<typename T>
inline QPMT_METHOD T qpmt<T>::get_qescale( int pmtid ) const 
{
    return pmtid < NUM_LPMT && pmtid > -1 ? lcqs[pmtid*2+1] : -2.f ; 
}
template<typename T>
inline QPMT_METHOD T qpmt<T>::get_lpmtcat_qe( int lpmtcat, T energy_eV ) const 
{
    return lpmtcat > -1 && lpmtcat < NUM_CAT ? qeshape_prop->interpolate( lpmtcat, energy_eV ) : -1.f ; 
}

template<typename T>
inline QPMT_METHOD void qpmt<T>::get_lpmtcat_stackspec( quad4& spec, int lpmtcat, T energy_eV ) const 
{
    const unsigned idx = lpmtcat*NUM_LAYR*NUM_PROP ; 
    const unsigned idx0 = idx + L0*NUM_PROP ; 
    const unsigned idx1 = idx + L1*NUM_PROP ; 
    const unsigned idx2 = idx + L2*NUM_PROP ; 

    spec.q0.f.x = rindex_prop->interpolate( idx0+0u, energy_eV ); 
    spec.q0.f.y = 0.f ; 
    spec.q0.f.z = 0.f ; 

    spec.q1.f.x = rindex_prop->interpolate( idx1+0u, energy_eV ); 
    spec.q1.f.y = rindex_prop->interpolate( idx1+1u, energy_eV ); 
    spec.q1.f.z = thickness[lpmtcat*NUM_LAYR+L1] ;

    spec.q2.f.x = rindex_prop->interpolate( idx2+0u, energy_eV ); 
    spec.q2.f.y = rindex_prop->interpolate( idx2+1u, energy_eV ); 
    spec.q2.f.z = thickness[lpmtcat*NUM_LAYR+L2] ;

    spec.q3.f.x = 1.f ;  // Vacuum RINDEX
    spec.q3.f.y = 0.f ; 
    spec.q3.f.z = 0.f ; 

    // dont zero .w as the pmtid info goes in there 
    //spec.q0.f.w = 0.f ; 
    //spec.q1.f.w = 0.f ; 
    //spec.q2.f.w = 0.f ; 
    //spec.q3.f.w = 0.f ; 
}


template<typename T>
inline QPMT_METHOD void qpmt<T>::get_lpmtid_stackspec( quad4& spec, int lpmtid, T energy_eV ) const 
{
    const int& lpmtcat = i_lcqs[lpmtid*2+0] ; 
    // printf("//qpmt::get_lpmtid_stackspec lpmtid %d lpmtcat %d \n", lpmtid, lpmtcat );  

    const T& qe_scale = lcqs[lpmtid*2+1] ; 
    const T qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ; 
    const T qe = qe_scale*qe_shape ; 

    spec.q0.i.w = lpmtcat ; 
    spec.q1.f.w = qe_scale ; 
    spec.q2.f.w = qe_shape ; 
    spec.q3.f.w = qe ; 

    get_lpmtcat_stackspec( spec, lpmtcat, energy_eV ); 
}

#ifdef WITH_CUSTOM4
template<>
inline QPMT_METHOD void qpmt<float>::get_lpmtid_ARTE( 
    float4& ARTE, 
    int lpmtid, 
    float wavelength_nm, 
    float minus_cos_theta, 
    float dot_pol_cross_mom_nrm ) const 
{
    const float energy_eV = hc_eVnm/wavelength_nm ; 

    quad4 spec ; 
    get_lpmtid_stackspec( spec, lpmtid, energy_eV ); 

    const float* ss = spec.cdata() ;
    const float& _qe = ss[15] ;

    Stack<float,4> stack ;  

    if( minus_cos_theta < zero )
    {
        stack.calc(wavelength_nm, -1.f, 0.f, ss, 16u );
        ARTE.w = _qe/stack.art.A ; 
    }
    else
    {
        ARTE.w = 0.f ;  
    }

    stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );
   
    const float& A = stack.art.A ; 
    const float& R = stack.art.R ; 
    const float& T = stack.art.T ; 
 
    ARTE.x = A ;         // aka theAbsorption
    ARTE.y = R/(1.f-A) ; // aka theReflectivity
    ARTE.z = T/(1.f-A) ; // aka theTransmittance
}
#endif

#endif



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template struct QUDARAP_API qpmt<float>;
//template struct QUDARAP_API qpmt<double>;
#endif



