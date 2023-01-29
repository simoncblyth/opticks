#pragma once
/**
slog.h
========

See notes/issues/logging_from_headeronly_impls.rst


**/

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>

#include "plog/Severity.h"
#include <plog/Log.h>

using plog::fatal ; 
using plog::error ; 
using plog::warning ; 
using plog::info ; 
using plog::debug ; 
using plog::verbose ; 


struct slog
{
    static plog::Severity envlevel( const char* key, const char* fallback); 
    static std::string Desc(plog::Severity level) ; 
    static std::string Dump() ; 
};

inline plog::Severity slog::envlevel(const char* key, const char* fallback)
{
    const char* val = getenv(key);
    const char* level = val ? val : fallback ; 
    plog::Severity severity = plog::severityFromString(level) ;

    if(strcmp(level, fallback) != 0)
    {
        std::cerr 
            << "slog::envlevel"
            << " adjusting loglevel by envvar  "
            << " key " << key  
            << " val " << val  
            << " fallback " << fallback
            << " level " << level
            << " severity " << severity 
            << std::endl 
            ;     
    }
    return severity ; 
}

inline std::string slog::Desc(plog::Severity level) // static
{
    std::stringstream ss ; 
    ss << "slog::Desc"
       << " level:" << level 
       << " plog::severityToString(level):" << plog::severityToString(level) 
       << std::endl  
       ; 
    std::string str = ss.str(); 
    return str ; 
}
inline std::string slog::Dump() // static
{
    std::stringstream ss ; 
    ss
        << " slog::dump " << std::endl
        << " plog::none    " << plog::none << std::endl 
        << " plog::fatal   " << plog::fatal << std::endl 
        << " plog::error   " << plog::error << std::endl   
        << " plog::warning " << plog::warning << std::endl 
        << " plog::info    " << plog::info << std::endl 
        << " plog::debug   " << plog::debug << std::endl 
        << " plog::verbose " << plog::verbose << std::endl   
        ; 

    std::string str = ss.str(); 
    return str ; 
}



