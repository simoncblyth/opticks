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
#include <iomanip>

#include "SPPM.hh"
#include "PLOG.hh"
#include "BFile.hh"
#include "NPY.hpp"
#include "ImageNPY.hpp"

const plog::Severity ImageNPY::LEVEL = PLOG::EnvLevel("ImageNPY", "INFO"); 


NPY<unsigned char>* ImageNPY::LoadPPM(const char* path, const bool yflip, const unsigned ncomp)  // static
{
    unsigned width(0) ; 
    unsigned height(0) ; 
    unsigned mode(0) ; 
    unsigned bits(0) ; 

    int rc0 = SPPM::readHeader(path, width, height, mode, bits ); 
    assert( rc0 == 0 && mode == 6 && bits == 255 ); 

    LOG(LEVEL) 
        << " path " << path 
        << " width " << width 
        << " height " << height 
        << " mode " << mode 
        << " bits " << bits 
        ;

    NPY<unsigned char>* img = NPY<unsigned char>::make( height, width, ncomp ) ;  
    img->zero(); 

    std::vector<unsigned char>& imgvec = img->vector();    

    int rc = SPPM::read(path, imgvec, width, height, ncomp, yflip ); 
    assert( rc == 0 );  

    return img ; 
}




void ImageNPY::SavePPM(const char* dir, const char* name,  const NPY<unsigned char>* a)
{
    bool createdir = true ; 
    std::string path = BFile::preparePath(dir, name, createdir); 
    SavePPMImp(path.c_str(), a ); 
}
void ImageNPY::SavePPM(const char* path_,  const NPY<unsigned char>* a)
{
    bool createdir = true ; 
    std::string path = BFile::preparePath(path_, createdir); 
    SavePPMImp(path.c_str(), a ); 
}

void ImageNPY::SavePPMImp(const char* path, const NPY<unsigned char>* a )
{
    unsigned nd = a->getDimensions(); 
    assert( nd == 3 ); 
    unsigned height = a->getShape(0); 
    unsigned width = a->getShape(1); 
    unsigned ncomp = a->getShape(2); 

    LOG(LEVEL) 
        << " path " << path 
        << " width " << width 
        << " height " << height 
        << " ncomp " << ncomp 
        ;

    bool yflip = false ; 
    const unsigned char* data = a->getValuesConst() ; 
    LOG(LEVEL) << " write to " << path ; 
    SPPM::write(path, data , width, height, ncomp, yflip );
} 





