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


#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "Opticks.hh"
#include "GGeo.hh"

#include "GPho.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    Opticks ok(argc, argv);
    GGeo* gg = GGeo::Load(&ok); 


    //const char* default_opt = "nidx,nrpo,post,lpst,ldrw,lpow,flgs" ; 
    const char* default_opt = "nidx,nrpo,post,okfl" ; 
    const char* opt = argc > 1 ? argv[1] : default_opt ;  

    const char* default_ox = "$TMP/G4OKTest/evt/g4live/natural/1/ox.npy" ; 
    const char* default_wy = "$TMP/G4OKTest/evt/g4live/natural/1/wy.npy" ; 
    const char* ox_path = argc > 2 ? argv[2] : default_ox ;  
    const char* wy_path = argc > 3 ? argv[3] : default_wy ;  


    NPY<float>* ox = NPY<float>::load( ox_path ) ; 
    if(ox == NULL ) return 0 ; 

    NPY<float>* wy = NPY<float>::load( wy_path ) ; 
    ox->setAux(wy); 

    LOG(info) 
        << " ox_path " << ox_path 
        << " ox " << ox->getShapeString()
        ; 
    LOG(info) 
        << " wy_path " << wy_path 
        << " wy " << ( wy ? wy->getShapeString() : "" )
        ; 


    GPho ph( gg, opt);   
    ph.setPhotons(ox);
    ph.setSelection('L');  // A:All L:Landed H:Hit  
    
    unsigned maxDump = 0 ; 
    ph.dump("GPhoTest", maxDump); 

    const char* ox_local = SStr::ReplaceEnd(ox_path, ".npy", "_local.npy" ); 
    ph.saveLocalPhotons( ox_local ); 

    return 0 ;
}

// om-;TEST=GPhoTest om-t
