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
#include "NPFold.h"

#include "qpmt.h"
#include "QProp.hh"
#include "QU.hh"
#include "QUDARAP_API_EXPORT.hh"

template<typename T>
struct QUDARAP_API QPMT
{
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

    QPMT(const NP* rindex, const NP* thickness, const NP* qeshape, const NP* lcqs);     

    void init(); 
    void init_thickness(); 
    void init_lcqs(); 

    static NP* MakeLookup_lpmtcat(int etype, unsigned num_domain ); 
    static NP* MakeLookup_lpmtid( int etype, unsigned num_domain, unsigned num_lpmtid ); 

    NPFold* get_fold() const ; 
    std::string desc() const ; 

    // .cc 
    void lpmtcat_check( int etype, const NP* domain, const NP* lookup) const ; 
    NP* lpmtcat_( int etype, const NP* domain) const ; 
    NP* lpmtid_(  int etype, const NP* domain, const NP* lpmtid) const ; 

    // inlines
    NP* lpmtcat_rindex(const NP* domain) const ; 
    NP* lpmtcat_qeshape(const NP* domain) const ; 
    NP* lpmtcat_stackspec( const NP* domain) const ; 

    NP* lpmtid_stackspec( const NP* domain, const NP* lpmtid ) const ; 
    NP* lpmtid_ART(  const NP* domain, const NP* lpmtid ) const ; 
    NP* lpmtid_ARTE( const NP* domain, const NP* lpmtid ) const ; 


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



template<typename T>
inline NP* QPMT<T>::MakeLookup_lpmtcat(int etype, unsigned num_domain )   // static
{
    const int& ni = qpmt<T>::NUM_CAT ; 
    const int& nj = qpmt<T>::NUM_LAYR ; 
    const int& nk = qpmt<T>::NUM_PROP ;  
    NP* lookup = nullptr ; 
    switch(etype)
    {
       case qpmt<T>::RINDEX:    lookup = NP::Make<T>( ni, nj, nk, num_domain ) ; break ; 
       case qpmt<T>::QESHAPE:   lookup = NP::Make<T>( ni,         num_domain ) ; break ; 
       case qpmt<T>::LPMTCAT_STACKSPEC: lookup = NP::Make<T>( ni, num_domain, 4, 4  )  ; break ; 
    }
    return lookup ; 
}


template<typename T>
inline NP* QPMT<T>::MakeLookup_lpmtid(int etype, unsigned num_domain, unsigned num_lpmtid )   // static
{
    const int ni = num_lpmtid ; 
    const int nj = num_domain ;
 
    NP* lookup = nullptr ; 
    switch(etype)
    {
       case qpmt<T>::LPMTID_STACKSPEC: lookup = NP::Make<T>( ni, nj, 4, 4  )  ; break ; 
       case qpmt<T>::LPMTID_ART:       lookup = NP::Make<T>( ni, nj, 4, 4  )  ; break ; 
       case qpmt<T>::LPMTID_ARTE:      lookup = NP::Make<T>( ni, nj, 4  )     ; break ; 
    }
    return lookup ; 
}


template<typename T>
inline NPFold* QPMT<T>::get_fold() const 
{
    NPFold* fold = new NPFold ; 

    fold->add("src_thickness", src_thickness ); 
    fold->add("src_rindex", src_rindex ); 
    fold->add("src_qeshape", src_qeshape ); 
    fold->add("src_lcqs", src_lcqs ); 

    fold->add("thickness", thickness ); 
    fold->add("rindex", rindex ); 
    fold->add("qeshape", qeshape ); 
    fold->add("lcqs", lcqs ); 

    fold->add("rindex_prop_a", rindex_prop->a ); 
    fold->add("qeshape_prop_a", qeshape_prop->a ); 

    return fold ; 
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

/**

QPMT::lpmtcat_rindex
QPMT::lpmtcat_qeshape
QPMT::lpmtcat_stackspec


**/


template<typename T>
inline NP* QPMT<T>::lpmtcat_rindex(const NP* domain) const { return lpmtcat_(qpmt<T>::RINDEX, domain) ; }

template<typename T>
inline NP* QPMT<T>::lpmtcat_qeshape(const NP* domain) const { return lpmtcat_(qpmt<T>::QESHAPE, domain) ; }

template<typename T>
inline NP* QPMT<T>::lpmtcat_stackspec(const NP* domain) const { return lpmtcat_(qpmt<T>::LPMTCAT_STACKSPEC, domain) ; }


template<typename T>
inline NP* QPMT<T>::lpmtid_stackspec(const NP* domain, const NP* lpmtid) const 
{ 
    return lpmtid_(qpmt<T>::LPMTID_STACKSPEC, domain, lpmtid) ; 
}
template<typename T>
inline NP* QPMT<T>::lpmtid_ART(const NP* domain, const NP* lpmtid) const 
{ 
    return lpmtid_(qpmt<T>::LPMTID_ART, domain, lpmtid) ; 
}
template<typename T>
inline NP* QPMT<T>::lpmtid_ARTE(const NP* domain, const NP* lpmtid) const 
{ 
    return lpmtid_(qpmt<T>::LPMTID_ARTE, domain, lpmtid) ; 
}


