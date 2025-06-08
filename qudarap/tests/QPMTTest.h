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
    static constexpr const char* LPMTID_LIST = "0,10,55,98,100,137,1000,10000,17611" ;

    const QPMT<T>* qpmt ;

    const char* lpmtid_list ;  // comma delimited string of ints from LPMTID_LIST envvar or default
    NP*         lpmtid ;       // array created from the string
    int         num_lpmtid ;

    NP*         lpmtcat ;      // array of cpu side category lookups using QPMT::get_lpmtcat
    int         num_lpmtcat ;

    NP*         energy_eV_domain ;
    NP*         theta_radians_domain ;   // from 0. to pi/2
    int         num_mct ;            // input from NUM_MCT envvar or default
    NP*         mct_domain ;         // from NP::MinusCosThetaLinearAngle

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
    lpmtid_list(ssys::getenvvar("LPMTID_LIST", LPMTID_LIST)), // pick some lpmtid (<17612) in envvar
    lpmtid(NPX::FromString<int>(lpmtid_list,',')),            // create array from string
    num_lpmtid(lpmtid->shape[0]),
    lpmtcat(NP::Make<int>(num_lpmtid)),
    num_lpmtcat(qpmt->get_lpmtcat(lpmtcat->values<int>(),lpmtid->cvalues<int>(),num_lpmtid)), // CPU side lookups
    energy_eV_domain(NP::Linspace<T>(1.55,15.50,1550-155+1)),
    theta_radians_domain(NP::ThetaRadians<T>(91,0.5)),
    num_mct(ssys::getenvint("NUM_MCT",900)),   // 181
    mct_domain(NP::MakeWithType<T>(NP::MinusCosThetaLinearAngle<double>(num_mct)))
{
}

template<typename T>
inline NPFold* QPMTTest<T>::make_qscan() const
{
    NPFold* qscan = new NPFold ;

    qscan->add("energy_eV_domain", energy_eV_domain ) ;
    qscan->add("theta_radians_domain", theta_radians_domain ) ;
    qscan->add("mct_domain", mct_domain ) ;
    qscan->add("lpmtid",  lpmtid ) ;
    qscan->add("lpmtcat", lpmtcat ) ;

    qscan->add("lpmtcat_rindex",    qpmt->lpmtcat_(qpmt_RINDEX,  energy_eV_domain) ) ;
    qscan->add("lpmtcat_qeshape",   qpmt->lpmtcat_(qpmt_QESHAPE, energy_eV_domain) ) ;
    qscan->add("lpmtcat_cetheta",   qpmt->lpmtcat_(qpmt_CETHETA, theta_radians_domain) ) ;
    qscan->add("lpmtcat_stackspec", qpmt->lpmtcat_(qpmt_CATSPEC, energy_eV_domain) ) ;

    qscan->add("spec", qpmt->mct_lpmtid_(qpmt_SPEC, mct_domain, lpmtid) ) ;
    qscan->add("art" , qpmt->mct_lpmtid_(qpmt_ART , mct_domain, lpmtid) ) ;
    qscan->add("arte", qpmt->mct_lpmtid_(qpmt_ARTE, mct_domain, lpmtid) ) ;
    qscan->add("comp", qpmt->mct_lpmtid_(qpmt_COMP, mct_domain, lpmtid) ) ;
    qscan->add("ll",   qpmt->mct_lpmtid_(qpmt_LL  , mct_domain, lpmtid) ) ;

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


