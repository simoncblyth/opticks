#pragma once

/**
SPPM
======

Implementation of the minimal(and uncompressed) PPM image file format. 

* PPM uses 24 bits per pixel: 8 for red, 8 for green, 8 for blue.
* https://en.wikipedia.org/wiki/Netpbm_format


DevNotes
----------

* examples/UseOpticksGLFWSnap/UseOpticksGLFWSnap.cc
* /Developer/OptiX_380/SDK/primeMultiGpu/primeCommon.cpp

**/


#include <string>
#include "plog/Severity.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SPPM 
{
    static const plog::Severity LEVEL ; 

    SPPM(); 

    unsigned char* pixels ; 
    int pwidth ; 
    int pheight ; 
    int pscale ; 
    bool yflip ; 

    std::string desc() const ; 

    virtual void download() = 0 ;

    void resize(int width, int height, int scale=1);
    void save(const char* path=NULL);
    void snap(const char* path=NULL); 

    static void save( const char* path, int width, int height, const unsigned char* image, bool yflip ) ;

    static void write( const char* filename, const unsigned char* image, int width, int height, int ncomp ) ;
    static void write( const char* filename, const         float* image, int width, int height, int ncomp ) ;

};




