#pragma once
/**
QPMT.hh : projecting PMT properties onto device using qpmt.h
==============================================================

* narrowing (or widening, or copying) inputs to template type done in ctor

* QPMT.hh does not directly depend on the JUNO specific JPMT.h, instead 
  the QPMT ctor accepts the arrays that JPMT.h collects from property files

  * hence JPMT.h (via PMTSim) only used in tests qudarap/tests/QPMTTest.cc not in qudarap

* QPMT.hh/qpmt.h layout has some similarities to QCerenkov.hh/qcerenkov.h 

**/

#include "plog/Severity.h"
#include "NP.hh"

#include "qpmt.h"
#include "QProp.hh"
#include "QU.hh"
#include "QUDARAP_API_EXPORT.hh"

template<typename T>
struct QUDARAP_API QPMT
{
    enum { RINDEX, QESHAPE } ; 

    static const plog::Severity LEVEL ; 

    const NP* src_rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, NEN, 2:[energy,value] )
    const NP* src_thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value )  
    const NP* src_qeshape ;   // (NUM_PMTCAT, NEN_SAMPLES~44, 2:[energy,value] )
    const NP* src_lcqs ;      // (NUM_LPMT, 2:[cat,qescale])

    const NP* rindex3 ;       // (NUM_PMTCAT*NUM_LAYER*NUM_PROP,  NEN, 2:[energy,value] )
    const NP* rindex ;  
    const QProp<T>* rindex_prop ; 

    const NP* qeshape ; 
    const QProp<T>* qeshape_prop ; 
   
    const NP* thickness ;  
    const NP* lcqs ;  


    qpmt<T>* pmt ; 
    qpmt<T>* d_pmt ; 

    QPMT( const NP* rindex, const NP* thickness, const NP* qeshape, const NP* lcqs );     

    void init(); 
    static NP* MakeLookup(int etype, unsigned domain_width ); 

    void save(const char* base) const ; 
    std::string desc() const ; 

    NP* interpolate(int etype, const NP* domain) const ; 
    NP* rindex_interpolate(const NP* domain) const ; 
    NP* qeshape_interpolate(const NP* domain) const ; 


};

/**
QPMT::QPMT
------------

1. copy rindex_ into 3D in rindex3 then narrows rindex3 into rindex, 
   NB this order preserves last prop column integer annotations
2. creates rindex_prop from rindex
3. narrows src_qeshape into qeshape
4. created qeshape_prop from qeshape
5. narrows src_thickness into thickness
6. narrows src_lcqs into lcqs

**/

template<typename T>
inline QPMT<T>::QPMT(const NP* rindex_ , const NP* thickness_, const NP* qeshape_, const NP* lcqs_ )
    :
    src_rindex(rindex_),
    src_thickness(thickness_),
    src_qeshape(qeshape_),
    src_lcqs(lcqs_),
    rindex3(  NP::MakeCopy3D(src_rindex)),   // make copy and change shape to 3D
    rindex(   NP::MakeWithType<T>(rindex3)), // adopt template type, potentially narrowing
    rindex_prop(new QProp<T>(rindex)),  
    qeshape(   NP::MakeWithType<T>(src_qeshape)), // adopt template type, potentially narrowing
    qeshape_prop(new QProp<T>(qeshape)),  
    thickness(NP::MakeWithType<T>(src_thickness)),
    lcqs(src_lcqs ? NP::MakeWithType<T>(src_lcqs) : nullptr),
    pmt(new qpmt<T>()),                    // hostside qpmt.h instance 
    d_pmt(nullptr)                         // devices pointer set in init
{
    init(); 
}

/**
QPMT::init
------------

1. populate hostside qpmt.h instance with device side pointers 
2. upload the hostside qpmt.h instance to GPU

**/


template<typename T>
inline void QPMT<T>::init()
{
    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ; 

    assert( src_rindex->has_shape(ni, nj, nk, -1, 2 )); 
    assert( src_thickness->has_shape(ni, nj, 1 )); 

    pmt->rindex_prop = rindex_prop->getDevicePtr() ;  
    pmt->qeshape_prop = qeshape_prop->getDevicePtr() ;  
    pmt->thickness = QU::UploadArray<T>(thickness->cvalues<T>(), thickness->num_values() ); 
    pmt->lcqs = lcqs ? QU::UploadArray<T>(lcqs->cvalues<T>(), lcqs->num_values() ) : nullptr ; 

    d_pmt = QU::UploadArray<qpmt<T>>( (const qpmt<T>*)pmt, 1u ) ;  
    // getting above line to link required template instanciation at tail of qpmt.h 
}


template<typename T>
inline NP* QPMT<T>::MakeLookup(int etype, unsigned domain_width )   // static
{
    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ;  
    NP* lookup = nullptr ; 
    switch(etype)
    {
       case RINDEX:  lookup = NP::Make<T>( ni, nj, nk, domain_width ) ; break ; 
       case QESHAPE: lookup = NP::Make<T>( ni,         domain_width ) ; break ; 
    }
    return lookup ; 
}




template<typename T>
inline void QPMT<T>::save(const char* base) const 
{
    src_thickness->save(base, "src_thickness.npy" );  
    src_rindex->save(base, "src_rindex.npy" );  
    src_qeshape->save(base, "src_qeshape.npy" );  
    if(src_lcqs) src_lcqs->save(base, "src_lcqs.npy" );  

    thickness->save(base, "thickness.npy" );  
    qeshape->save(base, "qeshape.npy" );  
    if(lcqs) lcqs->save(base, "lcqs.npy" );  

    rindex3->save(base, "rindex3.npy" );  
    rindex->save(base, "rindex.npy" );  
    rindex_prop->a->save(base, "rindex_prop_a.npy" );  

    qeshape_prop->a->save(base, "qeshape_prop_a.npy" );  

}

template<typename T>
inline std::string QPMT<T>::desc() const 
{
    int w = 30 ; 
    std::stringstream ss ; 
    ss << "QPMT::desc"
       << std::endl
       << std::setw(w) << "rindex "    << rindex->sstr() << std::endl
       << std::setw(w) << "qeshape " << qeshape->sstr() << std::endl
       << std::setw(w) << "thickness " << thickness->sstr() << std::endl
       << std::setw(w) << "lcqs " << lcqs->sstr() << std::endl
       << std::setw(w) << " pmt.rindex_prop " << pmt->rindex_prop  << std::endl 
       << std::setw(w) << " pmt.qeshape_prop " << pmt->qeshape_prop  << std::endl 
       << std::setw(w) << " pmt.thickness " << pmt->thickness  << std::endl 
       << std::setw(w) << " pmt.lcqs " << pmt->lcqs  << std::endl 
       << std::setw(w) << " d_pmt " << d_pmt   << std::endl 
       ;
    std::string s = ss.str(); 
    return s ;
}

template<typename T>
inline NP* QPMT<T>::rindex_interpolate(const NP* domain) const { return interpolate(RINDEX, domain) ; }

template<typename T>
inline NP* QPMT<T>::qeshape_interpolate(const NP* domain) const { return interpolate(QESHAPE, domain) ; }



