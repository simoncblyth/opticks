#pragma once
/**
SPMTAccessor.h
===============

Provides access to JUNO PMT data during standalone
optical only testing WITH_CUSTOM4 and without j/PMTSim.
For example::

   ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
   ~/opticks/g4cx/tests/G4CXApp.h
   ~/opticks/u4/U4Physics.hh
   ~/opticks/u4/U4Physics.cc

Attempt to provide standalone access to JUNO PMT data
without depending on junosw, using SPMT.h which is
how the data is passed to QPMT.hh and onto the GPU
in qpmt.h

**/


#include <cstdlib>
#include "sstr.h"
#include "SPMT.h"

#ifdef WITH_CUSTOM4
#include "C4IPMTAccessor.h"
struct SPMTAccessor : public C4IPMTAccessor
#else
struct SPMTAccessor
#endif
{
    static constexpr const char* TYPENAME = "SPMTAccessor" ;
    static SPMTAccessor* Load(const char* path);
    SPMTAccessor(const SPMT* pmt );
    const SPMT* pmt ;
    bool VERBOSE ;

    //[C4IPMTAccessor protocol methods
    int         get_num_lpmt() const ;
    double      get_qescale( int pmtid ) const ;
    int         get_pmtcat( int pmtid ) const ;
    double      get_pmtid_qe( int pmtid, double energy_MeV ) const ;
    void        get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const ;
    const char* get_typename() const ;
    //]

    static std::string Desc(std::array<double, 16>& ss );
};


inline SPMTAccessor* SPMTAccessor::Load(const char* path)
{
    SPMT* pmt = SPMT::CreateFromJPMT(path);
    //if(pmt == nullptr) return nullptr ;

    SPMTAccessor* accessor = new SPMTAccessor(pmt);
    assert( accessor );
    return accessor ;
}

inline SPMTAccessor::SPMTAccessor( const SPMT* _pmt )
    :
    pmt(_pmt),
    VERBOSE(getenv("SPMTAccessor__VERBOSE") != nullptr)
{
}

inline int SPMTAccessor::get_num_lpmt() const
{
    assert(0);  // use s_pmt.h directly
    return s_pmt::NUM_CD_LPMT ;
}

inline double SPMTAccessor::get_qescale( int pmtid ) const
{
    float qs = pmt->get_qescale_from_lpmtid(pmtid);
    return qs ;
}
inline int SPMTAccessor::get_pmtcat( int pmtid ) const
{
    int cat = pmt->get_lpmtcat_from_lpmtid(pmtid) ;

    if(VERBOSE) std::cout
        << "SPMTAccessor::get_pmtcat"
        << " pmtid " << pmtid
        << " cat " << cat
        << std::endl
        ;

    return cat ;
}



/**
SPMTAccessor::get_pmtid_qe
----------------------------

From C4CustomART::doIt it is apparent that PMTAccessor which
SPMTAccessor aims to stand in for in standalone running
without j/PMTSim is inconsistent in its energy units.

+----------------------------+-------------+
|  Method                    | energy unit |
+============================+=============+
| PMTAccessor::get_pmtid_qe  | energy_MeV  |
+----------------------------+-------------+
| PMTAccessor::get_stackspec | energy_eV   |
+----------------------------+-------------+

**/

inline double SPMTAccessor::get_pmtid_qe( int lpmtid, double energy_MeV ) const
{
    int lpmtidx = s_pmt::lpmtidx_from_lpmtid(lpmtid);

    float energy_eV = energy_MeV*1e6 ;
    float qe = pmt->get_lpmtidx_qe(lpmtidx, energy_eV) ;

    if(VERBOSE) std::cout
        << "SPMTAccessor::get_pmtid_qe"
        << " lpmtid " << lpmtid
        << " lpmtidx " << lpmtidx
        << " energy_MeV " << std::scientific << energy_MeV
        << " energy_eV " << std::scientific << energy_eV
        << " qe " << std::scientific << qe
        << std::endl
        ;

    return qe ;
}

inline void SPMTAccessor::get_stackspec( std::array<double, 16>& spec, int pmtcat, double energy_eV_ ) const
{
    float energy_eV = energy_eV_ ;
    quad4 q_spec ;
    pmt->get_stackspec(q_spec, pmtcat, energy_eV);

    const float* qq = q_spec.cdata();
    for(int i=0 ; i < 16 ; i++) spec[i] = double(qq[i]) ;

#ifdef DEBUG
    if(VERBOSE) std::cout
        << "SPMTAccessor::get_stackspec"
        << " pmtcat " << pmtcat
        << " energy " << std::scientific << energy
        << std::endl
        << Desc(spec)
        << std::endl
        ;
#endif

}

inline std::string SPMTAccessor::Desc(std::array<double, 16>& spec ) // static
{
    std::stringstream ss ;
    ss << "SPMTAccessor::Desc" << std::endl ;
    for(int i=0 ; i < 16 ; i++) ss
        << ( i % 4 == 0 ? "\n" : " " )
        << std::setw(10) << std::fixed  << spec[i] << " "
        << ( i == 15 ? "\n" : " " )
        ;

    std::string str = ss.str() ;
    return str ;
}



inline const char* SPMTAccessor::get_typename() const
{
    return TYPENAME ;
}




