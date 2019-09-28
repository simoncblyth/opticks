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

//  ggv --gsrclib
//  ggv --gsrclib --debug
//

#include "Opticks.hh"

#include "GSource.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"

#include "OPTICKS_LOG.hh"
#include "GGEO_BODY.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv);
    ok->configure(); 

    GSourceLib* sl = new GSourceLib(ok);

    sl->generateBlackBodySample();

    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    sl->add(source);

    NPY<float>* buf = sl->createBuffer();

    buf->save("$TMP/ggeo/GSourceLibTest/gsrclib.npy");  // read by ana/planck.py 


    return 0 ; 
}

