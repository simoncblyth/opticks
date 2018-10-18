#pragma once

#include <string>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SPPM 
{
    SPPM(); 

    unsigned char* pixels ; 
    int pwidth ; 
    int pheight ; 
    int pscale ; 

    std::string desc() const ; 

    virtual void download() = 0 ;

    void resize(int width, int height, int scale=1);
    void save(const char* path=NULL);
    void snap(const char* path=NULL); 

    static void save( const char* path, int width, int height, const unsigned char* image ) ;

    static void write( const char* filename, const unsigned char* image, int width, int height, int ncomp ) ;
    static void write( const char* filename, const         float* image, int width, int height, int ncomp ) ;

};




