#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBitSet 
{
    static void Parse( bool* bits, unsigned num_bits,  const char* spec ); 
    static std::string Desc( bool* bits, unsigned num_bits, bool reverse ); 
};







