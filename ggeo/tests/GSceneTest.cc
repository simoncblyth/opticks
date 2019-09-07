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

/**

GSceneTest --okcore debug --gltfname hello.gltf

**/
#include <set>
#include <string>

#include "NScene.hpp"

#include "Opticks.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"

#include "GScene.hh"

#include "OPTICKS_LOG.hh"


struct GSceneTest
{
    Opticks* ok ;
    GGeo* gg ; 

    GSceneTest(Opticks* ok_) 
        : 
        ok(ok_),
        gg(new GGeo(ok))
    {
        LOG(error) << "loadFromCache" ;  
        gg->loadFromCache();
        LOG(error) << "loadAnalyticFromCache" ;  
        gg->loadAnalyticFromCache();
        LOG(error) << "dumpStats" ;  
        gg->dumpStats();
    }
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv, "--gltf 3");
    ok.configure();

    const char* base = ok.getSrcGLTFBase() ;
    const char* name = ok.getSrcGLTFName() ;
    const char* config = ok.getGLTFConfig() ;
    int gltf = ok.getGLTF();

    assert(gltf == 3);

    LOG(info) << argv[0]
              << " base " << base
              << " name " << name
              << " config " << config
              << " gltf " << gltf 
              ; 

    GSceneTest gst(&ok);
    assert(gst.gg);


    return 0 ; 
}


