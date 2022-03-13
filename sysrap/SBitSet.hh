#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBitSet 
{
    unsigned num_bits ; 
    bool*    bits ; 

    static void Parse(       unsigned num_bits, bool* bits      , const char* spec ); 
    static std::string Desc( unsigned num_bits, const bool* bits, bool reverse ); 
    static SBitSet* Create(unsigned num_bits, const char* spec); 

    SBitSet( unsigned num_bits ); 
    void set(bool all); 
    void parse(const char* spec); 

    bool is_set(unsigned pos) const ;

    virtual ~SBitSet(); 
    std::string desc() const ; 
};



