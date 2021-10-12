#pragma once

#include "SYSRAP_API_EXPORT.hh"
#include "json.hpp"
#include <iostream>
#include "plog/Severity.h"


/**
SMeta
=======

Lightweight metadata based on https://nlohmann.github.io/json/ 
For a more heavy weight approach (with more historical baggage) see brap/BMeta.hpp

**/

struct SYSRAP_API SMeta
{
   static SMeta* Load(const char* path); 
   static SMeta* Load(const char* dir, const char* name); 
   static const plog::Severity LEVEL ; 

   nlohmann::json js ;      
   void save(const char* dir, const char* reldir, const char* name) const ; 
   void save(const char* dir, const char* name) const ; 
   void save(const char* path) const ; 

};


inline std::ostream& operator<<(std::ostream& os, const SMeta& sm)
{
    os << sm.js ; 
    return os; 
}



