
#include <iostream>
#include <fstream>

#include "PLOG.hh"
#include "SPPM.hh"


/*

PPM uses 24 bits per pixel: 8 for red, 8 for green, 8 for blue.

  /Developer/OptiX_380/SDK/primeMultiGpu/primeCommon.cpp

  https://en.wikipedia.org/wiki/Netpbm_format

  opticks/examples/UseOpticksGLFWSnap/UseOpticksGLFWSnap.cc

*/


SPPM::SPPM()
    :   
    pixels(NULL),
    pwidth(0),
    pheight(0),
    pscale(0)
{
}

std::string SPPM::desc() const 
{
    std::stringstream ss ; 
    ss << " SPPM " 
       << " pwidth " << pwidth 
       << " pheight " << pheight
       << " pscale " << pscale
        ;
    return ss.str(); 
}

void SPPM::resize( int width, int height, int scale )
{ 
    bool changed_size = !(width == pwidth && height == pheight && scale == pscale) ; 
    if( pixels == NULL || changed_size )
    {   
        delete [] pixels ;
        pixels = NULL ; 
        pwidth = width ; 
        pheight = height ; 
        pscale = scale ; 
        int size = 4 * pwidth * pscale * pheight * pscale ;
        pixels = new unsigned char[size];
        LOG(fatal) << "creating resized pixels buffer " << desc() ; 
    }   
}

void SPPM::save(const char* path)
{
    if(path == NULL ) path = "/tmp/SPPM.ppm" ; 
    save(path, pwidth*pscale, pheight*pscale, pixels );
    LOG(fatal) 
        << " path " << path 
        << " desc " << desc()
        ; 
}

void SPPM::save(const char* path, int width, int height, const unsigned char* image) 
{
    FILE * fp;
    fp = fopen(path, "wb");

    int ncomp = 4;
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ;

    bool yflip = true ;  // flip vertically
 
    int y0 = yflip ? height-1 : 0 ; 
    int y1 = yflip ?        0 : height - 1 ; 
    int yd = yflip ?       -1 : +1  ; 

    for( int y=y0; y >= y1; y+=yd ) 
    {
        for( int x=0; x < width ; ++x )
        {
            *(data + (y*width+x)*3+0) = image[(y*width+x)*ncomp+0] ;
            *(data + (y*width+x)*3+1) = image[(y*width+x)*ncomp+1] ;
            *(data + (y*width+x)*3+2) = image[(y*width+x)*ncomp+2] ;
        }
    }
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);
    delete[] data;
}

void SPPM::snap(const char* path)
{
    download(); 
    save(path); 
}


void SPPM::write( const char* filename, const float* image, int width, int height, int ncomp )
{

    std::ofstream out( filename, std::ios::out | std::ios::binary );
    if( !out ) 
    {
        std::cerr << "Cannot open file " << filename << "'" << std::endl;
        return;
    }

    out << "P6\n" << width << " " << height << "\n255" << std::endl;

    for( int y=height-1; y >= 0; --y ) // flip vertically
    {   
        for( int x = 0; x < width*ncomp; ++x ) 
        {   
            float val = image[y*width*ncomp + x]; 
            unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val*255.0f );
            out.put( cval );
        }   
    }
    std::cout << "Wrote file " << filename << std::endl;
}



void SPPM::write( const char* filename, const unsigned char* image, int width, int height, int ncomp )
{
    FILE * fp;
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int y=height-1; y >= 0; --y ) // flip vertically
    {   
        for( int x=0; x < width ; ++x ) 
        {   
            *(data + (y*width+x)*3+0) = image[(y*width+x)*ncomp+0] ;   
            *(data + (y*width+x)*3+1) = image[(y*width+x)*ncomp+1] ;   
            *(data + (y*width+x)*3+2) = image[(y*width+x)*ncomp+2] ;   
        }
    } 
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    std::cout << "Wrote file " << filename << std::endl;
    delete[] data;
}



