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

#include "sproc.h"
#include "qpmt_enum.h"
#include "qpmt.h"
#include "QProp.hh"

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
#else
#include "QU.hh"
#endif

#include "QUDARAP_API_EXPORT.hh"

template<typename T>
struct QUDARAP_API QPMT
{
    static const plog::Severity LEVEL ;
    static const QPMT<T>*    INSTANCE ;
    static const QPMT<T>*    Get();

    static std::string Desc();

    const char* ExecutableName ;

    const NP* src_rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, NEN, 2:[energy,value] )
    const NP* src_thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value )
    const NP* src_qeshape ;   // (NUM_PMTCAT, NEN_SAMPLES~44, 2:[energy,value] )
    const NP* src_cetheta ;   // (NUM_PMTCAT, NEN_SAMPLES~44, 2:[theta,value] )
    const NP* src_lcqs ;      // (NUM_LPMT, 2:[cat,qescale])

    const NP* rindex3 ;       // (NUM_PMTCAT*NUM_LAYER*NUM_PROP,  NEN, 2:[energy,value] )
    const NP* rindex ;
    const QProp<T>* rindex_prop ;

    const NP* qeshape ;
    const QProp<T>* qeshape_prop ;

    const NP* cetheta ;
    const QProp<T>* cetheta_prop ;


    const NP* thickness ;
    const NP* lcqs ;
    const int* i_lcqs ;  // CPU side lpmtid -> lpmtcat 0/1/2

    qpmt<T>* pmt ;
    qpmt<T>* d_pmt ;

    // .h
    QPMT(const NPFold* pf);

    // .cc
    void init();
    void init_prop();
    void init_thickness();
    void init_lcqs();

    // .h
    NPFold* serialize() const ;  // formerly get_fold
    std::string desc() const ;

    // .h : CPU side lpmtcat lookups
    int  get_lpmtcat( int lpmtid ) const ;
    int  get_lpmtcat( int* lpmtcat, const int* lpmtid , int num ) const ;

    static NP* MakeArray_lpmtcat(int etype, unsigned num_domain );
    static NP* MakeArray_lpmtid( int etype, unsigned num_domain, unsigned num_lpmtid );

    // .cc
    void lpmtcat_check_domain_lookup_shape( int etype, const NP* domain, const NP* lookup) const ;

    static const T* Upload(const NP* arr, const char* label);
    static T* Alloc(NP* out, const char* label);

    NP*  lpmtcat_( int etype, const NP* domain) const ;
    NP*  mct_lpmtid_(  int etype, const NP* domain, const NP* lpmtid) const ;

};


/**
QPMT::QPMT
------------

1. copy rindex_ into 3D in rindex3 then narrows rindex3 into rindex,
   NB this order preserves last prop column integer annotations
2. creates rindex_prop from rindex
3. narrows src_qeshape into qeshape
4. creates qeshape_prop from qeshape
5. creates cetheta_prop from cetheta
5. narrows src_thickness into thickness
6. narrows src_lcqs into lcqs

NB jpmt is the fold from SPMT::serialize not the raw fold from _PMTSimParamData

**/

template<typename T>
inline QPMT<T>::QPMT(const NPFold* jpmt )
    :
    ExecutableName(sproc::ExecutableName()),
    src_rindex(   jpmt->get("rindex")),
    src_thickness(jpmt->get("thickness")),
    src_qeshape(  jpmt->get("qeshape")),
    src_cetheta(  jpmt->get("cetheta")),
    src_lcqs(     jpmt->get_optional("lcqs")),
    rindex3(  NP::MakeCopy3D(src_rindex)),   // make copy and change shape to 3D
    rindex(   NP::MakeWithType<T>(rindex3)), // adopt template type, potentially narrowing
    rindex_prop(new QProp<T>(rindex)),
    qeshape(   NP::MakeWithType<T>(src_qeshape)), // adopt template type, potentially narrowing
    qeshape_prop(new QProp<T>(qeshape)),
    cetheta(   NP::MakeWithType<T>(src_cetheta)), // adopt template type, potentially narrowing
    cetheta_prop(new QProp<T>(cetheta)),
    thickness(NP::MakeWithType<T>(src_thickness)),
    lcqs(src_lcqs ? NP::MakeWithType<T>(src_lcqs) : nullptr),
    i_lcqs( lcqs ? (int*)lcqs->cvalues<T>() : nullptr ),    // CPU side lookup lpmtid->lpmtcat 0/1/2
    pmt(new qpmt<T>()),                    // host-side qpmt.h instance
    d_pmt(nullptr)                         // device-side pointer set at upload in init
{
    init();
}

// init in .cc
template<typename T>
inline NPFold* QPMT<T>::serialize() const  // formerly get_fold
{
    NPFold* fold = new NPFold ;

    fold->add("src_thickness", src_thickness );
    fold->add("src_rindex", src_rindex );
    fold->add("src_qeshape", src_qeshape );
    fold->add("src_cetheta", src_cetheta );
    fold->add("src_lcqs", src_lcqs );

    fold->add("thickness", thickness );
    fold->add("rindex", rindex );
    fold->add("qeshape", qeshape );
    fold->add("cetheta", cetheta );
    fold->add("lcqs", lcqs );

    fold->add("rindex_prop_a", rindex_prop->a );
    fold->add("qeshape_prop_a", qeshape_prop->a );
    fold->add("cetheta_prop_a", cetheta_prop->a );

    return fold ;
}

template<typename T>
inline std::string QPMT<T>::desc() const
{
    int w = 30 ;
    std::stringstream ss ;
    ss
       << "QPMT::desc"
       << std::endl
       << std::setw(w) << "rindex "    << rindex->sstr() << std::endl
       << std::setw(w) << "qeshape " << qeshape->sstr() << std::endl
       << std::setw(w) << "cetheta " << cetheta->sstr() << std::endl
       << std::setw(w) << "thickness " << thickness->sstr() << std::endl
       << std::setw(w) << "lcqs " << lcqs->sstr() << std::endl
       << std::setw(w) << " pmt.rindex_prop " << pmt->rindex_prop  << std::endl
       << std::setw(w) << " pmt.qeshape_prop " << pmt->qeshape_prop  << std::endl
       << std::setw(w) << " pmt.cetheta_prop " << pmt->cetheta_prop  << std::endl
       << std::setw(w) << " pmt.thickness " << pmt->thickness  << std::endl
       << std::setw(w) << " pmt.lcqs " << pmt->lcqs  << std::endl
       << std::setw(w) << " d_pmt " << d_pmt   << std::endl
       ;
    std::string s = ss.str();
    return s ;
}

/**
QPMT::get_lpmtcat
------------------

CPU side lookup of lpmtcat from lpmtid using i_lcqs array.

**/

template<typename T>
inline int QPMT<T>::get_lpmtcat( int lpmtid ) const
{
    assert( lpmtid > -1 && lpmtid < qpmt_NUM_LPMT );
    const int& lpmtcat = i_lcqs[lpmtid*2+0] ;
    return lpmtcat ;
}
template<typename T>
inline int QPMT<T>::get_lpmtcat( int* lpmtcat_, const int* lpmtid_, int num_lpmtid ) const
{
    for(int i=0 ; i < num_lpmtid ; i++)
    {
        int lpmtid = lpmtid_[i] ;
        int lpmtcat = get_lpmtcat(lpmtid) ;
        lpmtcat_[i] = lpmtcat ;
    }
    return num_lpmtid ;
}


/**
QPMT::MakeArray_lpmtcat
-------------------------

HMM: this is mainly for testing, perhaps put in QPMTTest ?

**/

template<typename T>
inline NP* QPMT<T>::MakeArray_lpmtcat(int etype, unsigned num_domain )   // static
{
    const int& ni = qpmt_NUM_CAT ;
    const int& nj = qpmt_NUM_LAYR ;
    const int& nk = qpmt_NUM_PROP ;
    NP* lookup = nullptr ;
    switch(etype)
    {
       case qpmt_RINDEX:  lookup = NP::Make<T>( ni, nj, nk, num_domain ) ; break ;
       case qpmt_QESHAPE: lookup = NP::Make<T>( ni,         num_domain ) ; break ;
       case qpmt_CETHETA: lookup = NP::Make<T>( ni,         num_domain ) ; break ;
       case qpmt_CATSPEC: lookup = NP::Make<T>( ni, num_domain, 4, 4  )  ; break ;
    }
    return lookup ;
}


template<typename T>
inline NP* QPMT<T>::MakeArray_lpmtid(int etype, unsigned num_domain, unsigned num_lpmtid )   // static
{
    const int ni = num_lpmtid ;
    const int nj = num_domain ;

    NP* lookup = nullptr ;
    switch(etype)
    {
       case qpmt_SPEC: lookup = NP::Make<T>( ni, nj, 4, 4  )       ; break ;
       case qpmt_ART:  lookup = NP::Make<T>( ni, nj, 4, 4  )       ; break ;
       case qpmt_COMP: lookup = NP::Make<T>( ni, nj, 1, 4, 4, 2 )  ; break ;
       case qpmt_LL:   lookup = NP::Make<T>( ni, nj, 4, 4, 4, 2 )  ; break ;
       case qpmt_ARTE: lookup = NP::Make<T>( ni, nj, 4  )          ; break ;
    }
    return lookup ;
}



