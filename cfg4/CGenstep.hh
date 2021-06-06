#pragma once
#include <string>
#include <boost/dynamic_bitset_fwd.hpp> 
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CGenstep
{
    unsigned index ;    // 0-based index of genstep in the event 
    unsigned photons ;  // number of photons in the genstep
    unsigned offset ;   // photon offset in the sequence of gensteps, ie number of photons in event before this genstep
    char     gentype ;  // 'C' 'S' 'T'
    boost::dynamic_bitset<>* mask ;


    CGenstep();
    CGenstep( unsigned index_ , unsigned photons_, unsigned offset_, char gentype_ );
    virtual ~CGenstep(); 

    void set(unsigned ix);   // ix: 0-based index within the genstep, must be less than .photons  


    std::string desc(const char* msg=nullptr) const ; 
    bool all() const ; 
    bool any() const ; 
    unsigned count() const ; 


    unsigned getGenflag() const ;  // SI CK TO from gentype 'C' 'S' 'T'

};





