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

    //[C4IPMTAccessor protocol methods
    int         get_num_lpmt() const ;
    double      get_pmtid_qe( int pmtid, double energy ) const ; 
    double      get_qescale( int pmtid ) const ; 
    int         get_pmtcat( int pmtid ) const ; 
    void        get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const ; 
    const char* get_typename() const ; 
    //]
}; 


inline SPMTAccessor* SPMTAccessor::Load(const char* path)
{
    SPMT* pmt = SPMT::Load(path); 
    if(pmt == nullptr) return nullptr ; 

    SPMTAccessor* accessor = new SPMTAccessor(pmt); 
    assert( accessor ); 
    return accessor ; 
}

inline SPMTAccessor::SPMTAccessor( const SPMT* _pmt ) 
    :
    pmt(_pmt)
{
}

inline int SPMTAccessor::get_num_lpmt() const 
{
    return SPMT::NUM_LPMT ; 
}
inline double SPMTAccessor::get_pmtid_qe( int pmtid, double energy ) const
{
    float energy_eV = energy ; 
    float qe = pmt->get_pmtid_qe(pmtid, energy_eV) ; 
    return qe ; 
}
inline double SPMTAccessor::get_qescale( int pmtid ) const 
{
    float qs = pmt->get_qescale(pmtid); 
    return qs ; 
}
inline int SPMTAccessor::get_pmtcat( int pmtid ) const 
{
    return pmt->get_lpmtcat(pmtid) ;   // assumes pmtid is for LPMT  
}

inline void SPMTAccessor::get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy ) const 
{
    float energy_eV = energy ; 
    quad4 spec ; 
    pmt->get_stackspec(spec, pmtcat, energy_eV); 

    const float* ff = spec.cdata(); 
    for(int i=0 ; i < 16 ; i++) ss[i] = double(ff[i]) ; 
}


inline const char* SPMTAccessor::get_typename() const 
{
    return TYPENAME ; 
}




