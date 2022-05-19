#pragma once

struct NPFold ; 

#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SSim
{
    static const plog::Severity LEVEL ; 
    static SSim* INSTANCE ; 
    static SSim* Get(); 
    static SSim* Load(const char* base); 

    NPFold* fold ; 

    SSim(); 

    void add(const char* k, const NP* a ); 
    const NP* get(const char* k) const ; 
    void load(const char* base); 
    void save(const char* base) const ; 

    std::string desc() const ; 
};







