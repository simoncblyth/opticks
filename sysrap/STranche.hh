#pragma once
#include "SYSRAP_API_EXPORT.hh"

/**
STranche
=========

Split *total* into tranches of mostly equal *max_tranche* sizes, 
except the remainder in the last tranche.

**/

struct SYSRAP_API STranche
{
    STranche(unsigned total_, unsigned max_tranche_) ;

    unsigned tranche_size(unsigned i) const ; 
    unsigned global_index(unsigned i, unsigned j) const ; 
    const char* desc() const ;
    void dump(const char* msg="STranche::dump");

    unsigned total ; 
    unsigned max_tranche ; 
    unsigned num_tranche ; 
    unsigned last_tranche ; 

};



