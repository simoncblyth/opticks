#pragma once
/**
QPMT.hh : projecting PMT properties onto device using qpmt.h
==============================================================

* narrowing (or widening, or copying) inputs to template type done in ctor

* QPMT.hh does not directly depend on the JUNO specific JPMT.h, instead 
  the QPMT ctor accepts the arrays that JPMT.h collects from property files

  * hence JPMT.h (via PMTSim) only used in tests qudarap/tests/QPMTTest.cc not in qudarap

* QPMT.hh/qpmt.h layout has some similarities to QCerenkov.hh/qcerenkov.h 


HMM: narrowing scrubs last column integer annotation which is essential for prop interpolation::

    In [7]: t.rindex[...,-1,-1].view(np.int32)
    Out[7]: 
    array([[[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]],

           [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]],

           [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]]], dtype=int32)

    In [3]: t.src_rindex[...,-1,-1].view(np.int64)
    Out[3]: 
    array([[[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]],

           [[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]],

           [[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]]])

**/

#include "NP.hh"

#include "plog/Severity.h"
#include "qpmt.h"
#include "QProp.hh"
#include "QU.hh"
#include "QUDARAP_API_EXPORT.hh"


template<typename T>
struct QUDARAP_API QPMT
{
    static const plog::Severity LEVEL ; 

    const NP* src_rindex ;  
    const NP* src_thickness ;  
    const NP* rindex ;  
    const NP* thickness ;  
    const QProp<T>* rindex_prop ; 

    qpmt<T>* pmt ; 
    qpmt<T>* d_pmt ; 

    QPMT( const NP* rindex, const NP* thickness );     
    void init(); 
    void save(const char* base) const ; 
    std::string desc() const ; 

    NP* interpolate(const NP* domain) const ; 
    NP* interpolate() const ; 
};


template<typename T>
inline QPMT<T>::QPMT(const NP* rindex_ , const NP* thickness_ )
    :
    src_rindex(rindex_),
    src_thickness(thickness_),
    rindex(   NP::MakeWithType<T>(rindex_)),
    thickness(NP::MakeWithType<T>(thickness_)),
    rindex_prop(QProp<T>::Make3D(rindex)),  
    pmt(new qpmt<T>()),
    d_pmt(nullptr)
{
    init(); 
}

template<typename T>
inline void QPMT<T>::init()
{
    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ; 

    assert(    rindex->has_shape(ni, nj, nk, -1, 2 )); 
    assert( thickness->has_shape(ni, nj, 1 )); 

    pmt->rindex_prop = rindex_prop->getDevicePtr() ;  
    pmt->thickness = QU::UploadArray<T>(thickness->cvalues<T>(), thickness->num_values() ); 
    d_pmt = QU::UploadArray<qpmt<T>>( (const qpmt<T>*)pmt, 1u ) ;  
    // getting above line to link required template instanciation at tail of qpmt.h 
}

template<typename T>
inline void QPMT<T>::save(const char* base) const 
{
    src_rindex->save(base, "src_rindex.npy" );  
    src_thickness->save(base, "src_thickness.npy" );  
    rindex->save(base, "rindex.npy" );  
    thickness->save(base, "thickness.npy" );  
}

template<typename T>
inline std::string QPMT<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "QPMT::desc"
       << std::endl
       << "rindex"
       << std::endl
       << rindex->sstr()
       << std::endl
       << "thickness"
       << std::endl
       << thickness->sstr()
       << std::endl
       << " pmt.rindex_prop " << pmt->rindex_prop 
       << " pmt.thickness " << pmt->thickness 
       << " d_pmt " << d_pmt  
       ;
    std::string s = ss.str(); 
    return s ;
}


