#pragma once

#include <vector>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

struct BRAP_API BBnd
{
    static const char DELIM ; 
    static const char* DuplicateOuterMaterial( const char* boundary0 ) ; 
    static const char* Form(const char* omat_, const char* osur_, const char* isur_, const char* imat_);

    BBnd(const char* spec);
    std::string desc() const ;

    const char* omat ; 
    const char* osur ; 
    const char* isur ; 
    const char* imat ; 

    std::vector<std::string> elem ;      
};

#include "BRAP_TAIL.hh"





