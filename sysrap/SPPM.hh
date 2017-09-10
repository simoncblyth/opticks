#pragma once

#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SPPM {
    public:
    static void write( const char* filename, const unsigned char* image, int width, int height, int ncomp ) ;
    static void write( const char* filename, const         float* image, int width, int height, int ncomp ) ;
};




