#pragma once

#include <vector>
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBitSet 
{
    static SBitSet*   Create( unsigned num_bits, const char* ekey, const char* fallback ); 
    static SBitSet*   Create( unsigned num_bits, const char* spec); 
    static void        Parse( unsigned num_bits, bool* bits      , const char* spec ); 
    static std::string Desc(  unsigned num_bits, const bool* bits, bool reverse ); 


    unsigned    num_bits ; 
    bool*       bits ; 

    // metadata
    const char* label ; 
    const char* spec ; 


    SBitSet( unsigned num_bits ); 
    virtual ~SBitSet(); 

    void        set_label(const char* label); 
    void        set_spec( const char* spec); 

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



