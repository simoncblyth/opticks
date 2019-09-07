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

// op --ngunconfig

#include "Opticks.hh"

#include "NGunConfig.hpp"
#include "NPY.hpp"

#include "BLog.hh"
#include <iostream>

int main(int argc, char** argv)
{
    Opticks ok(argc, argv);

    NGunConfig* gc = new NGunConfig ; 
    gc->parse();

    std::string cachedir = ok.getObjectPath("CGDMLDetector", 0);

    NPY<float>* transforms = NPY<float>::load(cachedir.c_str(), "gtransforms.npy");
    if(!transforms)
    {
       LOG(fatal) << argv[0] << " FAILED TO LOAD TRANFORMS FROM " << cachedir ; 
       return 1 ; 
    }

    unsigned int frameIndex = gc->getFrame() ;

    if(frameIndex < transforms->getShape(0))
    {
        glm::mat4 frame = transforms->getMat4( frameIndex ); 
        gc->setFrameTransform(frame) ; 
    }
    else
    {
        std::cout << "frameIndex not found " << frameIndex << std::endl ; 
    }

    gc->Summary();


    return 0 ; 
}
