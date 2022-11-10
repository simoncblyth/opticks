#pragma once
/**
QPMT.hh
=========

* where to narrow ? 

* QPMT.hh does not depend on JPMT.h directly, as thats too JUNO specific, 
  instead make ctor args the arrays that JPMT.h collects to keep generality 

  * so just use JPMT.h (via PMTSim) in qudarap/tests/QPMTTest.cc not in qudarap

* rindex array scrunch done in QProp::Make3D could be done in JPMT.h ?

* QPMT.hh/qpmt.h has some similarities to QCerenkov.hh/qcerenkov.h 

**/


#include "NP.hh"

#include "qpmt.h"
#include "QProp.hh"
#include "QU.hh"
#include "QUDARAP_API_EXPORT.hh"


template<typename T>
struct QUDARAP_API QPMT
{
    const NP* rindex ;  
    const NP* thickness ;  
    const QProp<T>* rindex_prop ; 

    qpmt<T>* pmt ; 
    qpmt<T>* d_pmt ; 

    QPMT( const NP* rindex, const NP* thickness );     
    void init(); 

    std::string desc() const ; 
};


template<typename T>
inline QPMT<T>::QPMT( const NP* rindex_ , const NP* thickness_ )
    :
    rindex(   NP::MakeType<T>(rindex_)),
    thickness(NP::MakeType<T>(thickness_)),
    rindex_prop(QProp<T>::Make3D(rindex)),  
    pmt(new qpmt<T>()),
    d_pmt(nullptr)
{
    init(); 
}

template<typename T>
inline void QPMT<T>::init()
{
    pmt->rindex = rindex_prop->getDevicePtr() ;  
    pmt->thickness = QU::UploadArray<T>(thickness->cvalues<T>(), thickness->num_values() ); 
    d_pmt = QU::UploadArray<qpmt<T>>( (const qpmt<T>*)pmt, 1u ) ;  
    // getting above line to link required template instanciation at tail of qpmt.h 
}

template<typename T>
inline std::string QPMT<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "QPMT::desc"
       << std::endl
       << "rindex"
       << std::endl
       << rindex->desc()
       << std::endl
       << "thickness"
       << std::endl
       << thickness->desc()
       << std::endl
       ;
    std::string s = ss.str(); 
    return s ;
}



//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QPMT<float>;
template struct QUDARAP_API QPMT<double>;
//#pragma GCC diagnostic pop
 




