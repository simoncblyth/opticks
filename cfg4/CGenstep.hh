#pragma once
#include <string>
#include <boost/dynamic_bitset_fwd.hpp> 
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CGenstep
{
    unsigned index ; 
    unsigned photons ; 
    unsigned offset ; 
    char     gentype ; 

    boost::dynamic_bitset<>* mask ;

    CGenstep( unsigned index_ , unsigned photons_, unsigned offset_, char gentype_ );
    virtual ~CGenstep(); 

    unsigned getRecordId(unsigned index_, unsigned photon_id) const ;
    void     markRecordId(unsigned index_, unsigned photon_id) ; 

    void set(unsigned i);  

    std::string desc(const char* msg="CGenstep::desc") const ; 
    bool all() const ; 
    bool any() const ; 
    unsigned count() const ; 

};





