/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <iostream>
#include <fstream>

#include "PLOG.hh"
#include "SPPM.hh"

const plog::Severity SPPM::LEVEL = PLOG::EnvLevel("SPPM", "DEBUG")  ; 

SPPM::SPPM()
    :   
    pixels(NULL),
    pwidth(0),
    pheight(0),
    pscale(0),
    yflip(true)
{
}

std::string SPPM::desc() const 
{
    std::stringstream ss ; 
    ss << " SPPM " 
       << " pwidth " << pwidth 
       << " pheight " << pheight
       << " pscale " << pscale
       << " yflip " << yflip
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
        LOG(LEVEL) << "creating resized pixels buffer " << desc() ; 
    }   
}

void SPPM::save(const char* path)
{
    if(path == NULL ) path = "/tmp/SPPM.ppm" ; 
    save(path, pwidth*pscale, pheight*pscale, pixels, yflip );
    LOG(fatal) 
        << " path " << path 
        << " desc " << desc()
        ; 
}

void SPPM::save(const char* path, int width, int height, const unsigned char* image, bool yflip)
{
    //LOG(info) << "saving to " << path ; 
    std::cout << "SPPM::save " << path << std::endl ;  

    FILE * fp;
    fp = fopen(path, "wb");

    int ncomp = 4;
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

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
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);
    delete[] data;
}




void SPPM::snap(const char* path)
{
    download(); 
    save(path); 
}


void SPPM::write( const char* filename, const float* image, int width, int height, int ncomp, bool yflip )
{
    std::ofstream out( filename, std::ios::out | std::ios::binary );
    if( !out ) 
    {
        std::cerr << "Cannot open file " << filename << "'" << std::endl;
        return;
    }

    out << "P6\n" << width << " " << height << "\n255" << std::endl;

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x = 0; x < width*ncomp; ++x ) 
        {   
            float val = image[y*width*ncomp + x];    // double flip ?
            unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val*255.0f );
            out.put( cval );
        }   
    }
    LOG(LEVEL) << "Wrote file (float*)" << filename ;
}

/**
SPPM::write
-------------

Note the intermediary array. This allows four component image data to be written into 
PPM format which is by definition 3 component and also allows for vertical flipping
as conventions vary in this regard resulting in a common need to yflip upside down images.

**/

void SPPM::write( const char* filename, const unsigned char* image, int width, int height, int ncomp, bool yflip )
{
    FILE * fp;
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) 
    {   
        int y = yflip ? height - 1 - h : h ;   // flip vertically

        for( int x=0; x < width ; ++x ) 
        {   
            *(data + (y*width+x)*3+0) = image[(h*width+x)*ncomp+0] ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)*ncomp+1] ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)*ncomp+2] ;   
        }
    } 


    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    LOG(LEVEL) << "Wrote file (unsigned char*) " << filename  ;
    delete[] data;
}



/**
SPPM::read
------------

* https://en.wikipedia.org/wiki/Netpbm

**/

void SPPM::dumpHeader( const char* path )
{
    unsigned width(0); 
    unsigned height(0); 
    unsigned mode(0); 
    unsigned bits(0); 

    int rc = readHeader(path, width, height, mode, bits ); 

    LOG(info)
        << " path " << path 
        << " width " << width
        << " height " << height
        << " mode " << mode
        << " bits " << bits
        << " rc " << rc 
        ;
}

int SPPM::readHeader( const char* path, unsigned& width, unsigned& height, unsigned& mode, unsigned& bits )
{
    std::ifstream f(path, std::ios::binary);
    if(f.fail())
    {
        std::cout << "Could not open path: " << path << std::endl;
        return 1 ;
    }

    mode = 0;
    std::string s;
    f >> s;
    if (s == "P3") mode = 3;
    else if (s == "P6") mode = 6;
    
    f >> width ;
    f >> height ;
    f >> bits;

    f.close();
    return 0 ; 
}


int SPPM::read( const char* path, std::vector<unsigned char>& img, unsigned& width, unsigned& height, const unsigned ncomp, const bool yflip )
{
    std::ifstream f(path, std::ios::binary);
    if(f.fail())
    {
        std::cout << "Could not open path: " << path << std::endl;
        return 1 ;
    }

    int mode = 0;
    std::string s;
    f >> s;
    if (s == "P3") mode = 3;
    else if (s == "P6") mode = 6;
    assert( mode == 6 ); 
    
    f >> width ;
    f >> height ;

    int bits = 0;
    f >> bits;
    assert( bits == 255 ); 
    f.get();

    unsigned filesize = width*height*3 ; 
    unsigned arraysize = ncomp == 3 ? filesize : width*height*ncomp  ; 

    img.clear(); 
    img.resize(arraysize);
    unsigned char* imgdata = img.data(); 

    if( ncomp == 3 && yflip == false ) // slurp straight into the vector when no shuffling needed 
    {
        f.read((char*)imgdata, filesize);
    }
    else
    {
        unsigned char* tmp = new unsigned char[filesize] ; 
        f.read( (char*)tmp, filesize);

        for( int h=0; h < height ; h++ ) 
        {   
            int y = yflip ? height - 1 - h : h ;   // flip vertically
            for( int x=0; x < width ; ++x ) 
            {   
                for( int k=0 ; k < ncomp ; k++ )
                { 
                    imgdata[ (h*width+x)*ncomp + k] = k < 3 ? tmp[(y*width+x)*3+k] : 0xff ;        
                }
            }
        } 
        delete [] tmp ; 
    }
    f.close();
    return 0  ; 
}


