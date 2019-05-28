#pragma once
/**
SAbbrev
========

**/


#include <string>
#include <vector>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAbbrev
{
    SAbbrev( const std::vector<std::string>& names_ ); 
    void init(); 
    bool isFree(const std::string& ab) const ;
    void dump() const ; 
   
    const std::vector<std::string>& names ; 
    std::vector<std::string> abbrev ; 
};


