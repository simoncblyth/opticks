#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBitSet 
{
    static SBitSet*   Create( unsigned num_bits, const char* spec); 
    static void        Parse( unsigned num_bits, bool* bits      , const char* spec ); 
    static std::string Desc(  unsigned num_bits, const bool* bits, bool reverse ); 


    unsigned num_bits ; 
    bool*    bits ; 

    SBitSet( unsigned num_bits ); 
    virtual ~SBitSet(); 

    void        set(bool all); 
    void        parse(const char* spec); 
    bool        is_set(unsigned pos) const ;

    unsigned    count() const ; 
    bool        all() const ; 
    bool        any() const ; 
    bool        none() const ; 

    void get_pos( std::vector<unsigned>& pos ) const ; 

    std::string desc() const ; 
};



