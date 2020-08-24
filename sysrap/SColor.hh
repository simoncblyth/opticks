#pragma once

#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SColor 
{
    unsigned char r ; 
    unsigned char g ; 
    unsigned char b ; 
    unsigned char get(unsigned i) const ; 
};

struct SYSRAP_API SColors
{
    static const SColor red ; 
    static const SColor green ; 
    static const SColor blue ; 
    static const SColor cyan ; 
    static const SColor magenta ; 
    static const SColor yellow ; 
    static const SColor white ; 
    static const SColor black ; 
};
