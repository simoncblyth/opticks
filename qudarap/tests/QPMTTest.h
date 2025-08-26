#pragma once
/**
QPMTTest.h
=============

NB this header is used by three QPMT tests:

1. standard om built QPMTTest (CUDA)
2. standalone QPMT_Test.sh built QPMT_Test (CUDA)
3. standalone QPMT_MockTest.sh built QPMT_MockTest (MOCK_CURAND using CPU only)

**/

template <typename T> struct QPMT ;
struct NP ;
struct NPFold ;


template<typename T>
struct QPMTTest
{
    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611,50000,51000,52000,52399,52400,52747,54000,54004" ;
    // HMM: duplicates SPMT::LPMTID_LIST

    const QPMT<T>* qpmt ;

    const char* lpmtid_list ;  // comma delimited string of ints from LPMTID_LIST envvar or default

    NP*         lpmtid ;       // array created from the string
    int         num_lpmtid ;


    NP*         lpmtidx ;
    int         num_lpmtidx ;

    NP*         lpmtcat ;      // array of cpu side category lookups using QPMT::get_lpmtcat
    int         num_lpmtcat ;

    NP*         energy_eV_domain ;
    NP*         theta_radians_domain ;   // from 0. to pi/2
    NP*         costh_domain ;           // from 0. to 1.  (reverse:true)

    int         num_mct ;            // input from NUM_MCT envvar or default
    NP*         mct_domain ;         // from NP::MinusCosThetaLinearAngle

    // small PMT info
    static NP* GetSPMTID(const char* spec);
    //static constexpr const char* SPMTID_SPEC = "20000,30000,40000,45599" ; // 45599 + 1 - 20000 = 25600
    static constexpr const char* SPMTID_SPEC = "[20000:45600]" ; // 45599 + 1 - 20000 = 25600

    const char* spmtid_spec ;
    NP*         spmtid ;       // array created from the string
    int         num_spmtid ;
    NP*         spmtidx ;
    int         num_spmtidx ;


    QPMTTest(const NPFold* jpmt );

    NPFold* make_qscan() const ;
    NPFold* serialize(const char* scan_name=nullptr) const ;
};


#include "ssys.h"
#include "srng.h"

#include "NPX.h"
#include "NPFold.h"

#include "QPMT.hh"






/**
QPMTTest::QPMTTest
--------------------

np.linspace( 1.55, 15.50, 1550-155+1 )

**/

template<typename T>
inline QPMTTest<T>::QPMTTest(const NPFold* jpmt  )
    :
    qpmt(new QPMT<T>(jpmt)),
    lpmtid_list(ssys::getenvvar("LPMTID_LIST", LPMTID_LIST)), // pick some lpmtid
    lpmtid(NPX::FromString<int>(lpmtid_list,',')),            // create array from string
    num_lpmtid(lpmtid->shape[0]),
    lpmtidx(NP::Make<int>(num_lpmtid)),
    num_lpmtidx(qpmt->get_lpmtidx_from_lpmtid(lpmtidx->values<int>(),lpmtid->cvalues<int>(),num_lpmtid)), // CPU side lookups
    lpmtcat(NP::Make<int>(num_lpmtid)),
    num_lpmtcat(qpmt->get_lpmtcat_from_lpmtid(lpmtcat->values<int>(),lpmtid->cvalues<int>(),num_lpmtid)), // CPU side lookups
    energy_eV_domain(NP::Linspace<T>(1.55,15.50,1550-155+1)),
    theta_radians_domain(NP::ThetaRadians<T>(91,0.5)),
    costh_domain(NP::Cos(theta_radians_domain)),
    num_mct(ssys::getenvint("NUM_MCT",900)),   // 181
    mct_domain(NP::MakeWithType<T>(NP::MinusCosThetaLinearAngle<double>(num_mct))),
    spmtid_spec(ssys::getenvvar("SPMTID_SPEC", SPMTID_SPEC)),
    spmtid(GetSPMTID(spmtid_spec)),            // create array from string
    num_spmtid(spmtid->shape[0]),
    spmtidx(NP::Make<int>(num_spmtid)),
    num_spmtidx(qpmt->get_spmtidx_from_spmtid(spmtidx->values<int>(),spmtid->cvalues<int>(),num_spmtid)) // CPU side lookups
{
}

/**
QPMTTest::GetSPMTID
--------------------

Either comma delimited lists OR slice ARange_FromString spec are handled, eg::

    "20000,30000,40000,45599"   # 4 listed values
    "[20000:45600]"             # 25600 values using python slice spec (start,stop,step)

**/


template<typename T>
inline NP* QPMTTest<T>::GetSPMTID(const char* spec) // static
{
    if(!spec) return nullptr ;
    bool is_comma_delimited_list = strstr(spec,",");
    return  is_comma_delimited_list ? NPX::FromString<int>(spec,',') : NP::ARange_FromString<int>(spec) ;
}


template<typename T>
inline NPFold* QPMTTest<T>::make_qscan() const
{
    NPFold* qscan = new NPFold ;

    qscan->add("energy_eV_domain", energy_eV_domain ) ;
    qscan->add("theta_radians_domain", theta_radians_domain ) ;
    qscan->add("costh_domain", costh_domain ) ;
    qscan->add("mct_domain", mct_domain ) ;

    qscan->add("lpmtid",  lpmtid ) ;
    qscan->add("lpmtidx", lpmtidx ) ;
    qscan->add("lpmtcat", lpmtcat ) ;

    qscan->add("spmtid",  spmtid ) ;
    qscan->add("spmtidx", spmtidx ) ;


    qscan->add("pmtcat_rindex",    qpmt->pmtcat_scan(qpmt_RINDEX,    energy_eV_domain) ) ;
    qscan->add("pmtcat_stackspec", qpmt->pmtcat_scan(qpmt_CATSPEC,   energy_eV_domain) ) ;
    qscan->add("pmtcat_qeshape",   qpmt->pmtcat_scan(qpmt_QESHAPE,   energy_eV_domain) ) ;
    qscan->add("pmtcat_s_qeshape", qpmt->pmtcat_scan(qpmt_S_QESHAPE, energy_eV_domain) ) ;
    qscan->add("pmtcat_cetheta",   qpmt->pmtcat_scan(qpmt_CETHETA,   theta_radians_domain) ) ;
    qscan->add("pmtcat_cecosth",   qpmt->pmtcat_scan(qpmt_CECOSTH,   costh_domain ) ) ;

    qscan->add("spec",    qpmt->mct_lpmtid_scan(qpmt_SPEC,    mct_domain, lpmtid) ) ;
    qscan->add("spec_ce", qpmt->mct_lpmtid_scan(qpmt_SPEC_ce, mct_domain, lpmtid) ) ;

    qscan->add("art" ,    qpmt->mct_lpmtid_scan(qpmt_ART , mct_domain, lpmtid) ) ;
    qscan->add("arte",    qpmt->mct_lpmtid_scan(qpmt_ARTE, mct_domain, lpmtid) ) ;
    qscan->add("atqc",    qpmt->mct_lpmtid_scan(qpmt_ATQC, mct_domain, lpmtid) ) ;
    qscan->add("comp",    qpmt->mct_lpmtid_scan(qpmt_COMP, mct_domain, lpmtid) ) ;
    qscan->add("ll",      qpmt->mct_lpmtid_scan(qpmt_LL  , mct_domain, lpmtid) ) ;


    qscan->add("s_qescale",   qpmt->spmtid_scan(qpmt_S_QESCALE, spmtid ) ) ;


    return qscan ;
}

template<typename T>
inline NPFold* QPMTTest<T>::serialize(const char* scan_name_) const
{
    const char* scan_name = scan_name_ ? scan_name_ : "qscan" ;
    NPFold* f = new NPFold ;
    f->add_subfold("qpmt", qpmt->serialize() );
    f->add_subfold(scan_name, make_qscan() );
    return f ;
}


