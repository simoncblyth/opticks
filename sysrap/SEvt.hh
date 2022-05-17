#pragma once
/**
SEvt.hh
=========

Attempting to do this header only gives duplicate symbol for the SEvt::INSTANCE.
It seems c++17 would allow "static inline SEvt* INSTANCE"  but c++17 
is problematic on laptop, so use separate header and implementation
for simplicity. 

It is possible with c++11 but is non-intuitive

* https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library


Replacing cfg4/CGenstepCollector

HMM: gs vector of sgs provides summary of the full genstep, 
changing the first quad of the genstep to hold this summary info 
would avoid the need for the summary vector and mean the genstep 
index and photon offset info was available on device.

Header of full genstep has plenty of spare bits to squeeze in
index and photon offset in addition to  gentype/trackid/matline/numphotons 

**/

#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "plog/Severity.h"
#include "scuda.h"
#include "squad.h"
#include "sgs.h"


#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SEvt
{
    static const plog::Severity LEVEL ; 
    static SEvt* INSTANCE ; 
    static SEvt* Get() ; 
    static sgs AddGenstep(const quad6& q); 

    std::vector<quad6> genstep ; 
    std::vector<sgs>   gs ; 

    SEvt(); 

    void clear() ; 
    unsigned getNumGenstep() const ; 
    unsigned getNumPhoton() const ; 
    sgs addGenstep(const quad6& q) ; 
    void saveGenstep(const char* dir) const ; 

    std::string desc() const ; 
};



