#pragma once
/**
sppm.h
=======

**/

#include <cassert>
#include <fstream>
#include <iostream>


struct sppm
{
    static unsigned char* CreateThreeComponentImage( int width, int height, int ncomp, const unsigned char* image, bool yflip ); 
    static void Write(const char* path, int width, int height, int ncomp, const unsigned char* image, bool yflip); 
    static unsigned char* CreateImageData( int width, int height, int ncomp, bool yflip ); 
};


/**
sppm::CreateThreeComponentImage
---------------------------------

Creates 3 component *data* from 3 or 4 component *image* with optional yflip.

**/

inline unsigned char* sppm::CreateThreeComponentImage( int width, int height, int ncomp, const unsigned char* image, bool yflip )
{
    assert( ncomp == 3 || ncomp == 4 ); 
    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ;
    for( int h=0 ; h < height ; h++ ) 
    {
        int y = yflip ? height - 1 - h : h ;  

        for( int x=0; x < width ; ++x )
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)*ncomp+0] ;
            *(data + (y*width+x)*3+1) = image[(h*width+x)*ncomp+1] ;
            *(data + (y*width+x)*3+2) = image[(h*width+x)*ncomp+2] ;
        }
    }
    return data ; 
}

inline void sppm::Write(const char* path, int width, int height, int ncomp, const unsigned char* image, bool yflip)
{
    FILE * fp;
    fp = fopen(path, "wb");

    if(!fp) std::cout << "sppm::Write FAILED FOR [" << ( path ? path : "-" )  << std::endl ;  
    assert(fp); 
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned char* data = CreateThreeComponentImage( width, height, ncomp, image, yflip ); 
    size_t size_in_bytes = height*width*3*sizeof(unsigned char) ; 
    size_t count = 1 ; 
    fwrite(data, size_in_bytes, count , fp);
    fclose(fp);

    delete[] data;
}

inline unsigned char* sppm::CreateImageData( int width, int height, int ncomp, bool yflip )
{
    size_t size =  height*width*ncomp ; 
    unsigned char* data = new unsigned char[size] ;
    for( int h=0; h < height ; h++ ) 
    {   
        int y = yflip ? height - 1 - h : h ;   // flip vertically
        for( int x=0; x < width ; ++x ) 
        {   
            for( int k=0 ; k < ncomp ; k++ ) data[ (h*width+x)*ncomp + k] = k < 3 ? 0xaa : 0xff ;        
        }
    } 
    return data ; 
}

