#pragma once

#include <string>
#include <vector>

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GMaterialLib ; 
class GSurfaceLib ; 

struct GGEO_API GBnd
{
    GBnd(const char* spec_, bool flip_, GMaterialLib* mlib_, GSurfaceLib* slib_, bool dbgbnd_ ) ;
    void init(bool flip_);
    void check();

    unsigned           UNSET ; 
    const char*         spec ; 
    GMaterialLib*       mlib ;    // getIndex may trigger a close, so cannot be const 
    GSurfaceLib*        slib ; 
    bool                dbgbnd ; 


    const char* omat_ ; 
    const char* osur_ ; 
    const char* isur_ ; 
    const char* imat_ ;

    unsigned  omat ; 
    unsigned  osur ; 
    unsigned  isur ; 
    unsigned  imat ; 

    bool has_osur() const ; 
    bool has_isur() const ; 


    std::vector<std::string> elem ; 
};

#include "GGEO_TAIL.hh"

