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


DevNotes
----------

* examples/UseOpticksGLFWSnap/UseOpticksGLFWSnap.cc
* examples/UseOpticksGLFWSPPM/UseOpticksGLFWSPPM.cc
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

    static void write( const char* filename, const unsigned char* image, int width, int height, int ncomp, bool yflip) ;
    static void write( const char* filename, const         float* image, int width, int height, int ncomp, bool yflip) ;

};




