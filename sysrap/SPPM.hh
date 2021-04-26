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

#pragma once

/**
SPPM
======

Implementation of the minimal(and uncompressed) PPM image file format. 

* PPM uses 24 bits per pixel: 8 for red, 8 for green, 8 for blue.
* https://en.wikipedia.org/wiki/Netpbm_format

* SPPM is used as the base class of oglrap/Pix and used by oglra/Frame::snap 


Deficiencies
-------------

Much of SPPM is general image manipulation unrelated to the PPM image format. 

* TODO : integrate image manipulation into SIMG or perhspa SIMGExtra


DevNotes
----------

* examples/UseOpticksGLFWSnap/UseOpticksGLFWSnap.cc
* examples/UseOpticksGLFWSPPM/UseOpticksGLFWSPPM.cc
* /Developer/OptiX_380/SDK/primeMultiGpu/primeCommon.cpp

**/


#include <string>
#include <vector>
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

    static void write( const char* filename, const unsigned char* image, int width, int height, int ncomp, bool yflip) ;
    static void write( const char* filename, const         float* image, int width, int height, int ncomp, bool yflip) ;


    static int read( const char* path, std::vector<unsigned char>& data, unsigned& width, unsigned& height, const unsigned ncomp, const bool yflip );
    static void dumpHeader( const char* path ) ; 
    static int readHeader( const char* path, unsigned& width, unsigned& height, unsigned& mode, unsigned& bits ) ; 

    static unsigned char* MakeTestImage(const int width, const int height, const int ncomp, const bool yflip,  const char* config); 
    static unsigned ImageCompare(const int width, const int height, const int ncomp, const unsigned char* imgdata, const unsigned char* imgdata2 ); 

    static void AddBorder( std::vector<unsigned char>& img, const int width, const int height, const int ncomp, const bool yflip );
    static void AddBorder(unsigned char* imgdata, const int width, const int height, const int ncomp, const bool yflip );
    static void AddMidline( std::vector<unsigned char>& img, const int width, const int height, const int ncomp, const bool yflip );
    static void AddMidline(unsigned char* imgdata, const int width, const int height, const int ncomp, const bool yflip );
    static void AddQuadline( std::vector<unsigned char>& img, const int width, const int height, const int ncomp, const bool yflip );
    static void AddQuadline(unsigned char* imgdata, const int width, const int height, const int ncomp, const bool yflip );



    // hmm need an SImage ? or do in SPPM ?
 
};




