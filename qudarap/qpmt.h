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

template<typename T>
struct qpmt
{
    enum { NUM_CAT = 3, NUM_LAYR = 4, NUM_PROP = 2 } ;  
    enum { L0, L1, L2, L3 } ; 
    enum { RINDEX, KINDEX, QESHAPE, STACKSPEC } ; 

    qprop<T>* rindex_prop ;
    qprop<T>* qeshape_prop ;

    T*        thickness ; 
    T*        lcqs ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QPMT_METHOD void get_stackspec( quad4& spec, int cat, T energy_eV ); 
#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
template<typename T>
inline QPMT_METHOD void qpmt<T>::get_stackspec( quad4& spec, int cat, T energy_eV )
{
    const unsigned idx = cat*NUM_LAYR*NUM_PROP ; 
    const unsigned idx0 = idx + L0*NUM_PROP ; 
    const unsigned idx1 = idx + L1*NUM_PROP ; 
    const unsigned idx2 = idx + L2*NUM_PROP ; 

    spec.q0.f.x = rindex_prop->interpolate( idx0+0u, energy_eV ); 
    spec.q0.f.y = 0.f ; 
    spec.q0.f.z = 0.f ; 
    spec.q0.f.w = 0.f ; 

    spec.q1.f.x = rindex_prop->interpolate( idx1+0u, energy_eV ); 
    spec.q1.f.y = rindex_prop->interpolate( idx1+1u, energy_eV ); 
    spec.q1.f.z = thickness[cat*NUM_LAYR+L1] ;
    spec.q1.f.w = 0.f ; 

    spec.q2.f.x = rindex_prop->interpolate( idx2+0u, energy_eV ); 
    spec.q2.f.y = rindex_prop->interpolate( idx2+1u, energy_eV ); 
    spec.q2.f.z = thickness[cat*NUM_LAYR+L2] ;
    spec.q2.f.w = 0.f ; 

    spec.q3.f.x = 1.f ;  // Vacuum RINDEX
    spec.q3.f.y = 0.f ; 
    spec.q3.f.z = 0.f ; 
    spec.q3.f.w = 0.f ; 
}
#endif



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template struct QUDARAP_API qpmt<float>;
//template struct QUDARAP_API qpmt<double>;
#endif



