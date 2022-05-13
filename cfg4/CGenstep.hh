#pragma once
#include <string>

#ifdef WITH_CGENSTEP_MASK
#include <boost/dynamic_bitset_fwd.hpp> 
#endif

#include "CFG4_API_EXPORT.hh"

struct CFG4_API CGenstep
{
    unsigned index ;    // 0-based index of genstep in the event 
    unsigned photons ;  // number of photons in the genstep
    unsigned offset ;   // photon offset in the sequence of gensteps, ie number of photons in event before this genstep
    char     gentype ;  // 'C' 'S' 'T'

#ifdef WITH_CGENSTEP_MASK
    boost::dynamic_bitset<>* mask ;
#else
    char* mask ; 
#endif


    CGenstep();
    CGenstep( unsigned index_ , unsigned photons_, unsigned offset_, char gentype_ );
    std::string desc(const char* msg=nullptr) const ; 
    unsigned getGenflag() const ;  // SI CK TO from gentype 'C' 'S' 'T'

    virtual ~CGenstep(); 

#ifdef WITH_CGENSTEP_MASK
    void set(unsigned ix);   // ix: 0-based index within the genstep, must be less than .photons  
    bool all() const ; 
    bool any() const ; 
    unsigned count() const ; 
#endif


};





