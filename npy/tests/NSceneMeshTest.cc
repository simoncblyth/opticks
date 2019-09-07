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


#include "SSys.hh"
#include "BOpticksResource.hh"


#include "NGLTF.hpp"
#include "NScene.hpp"
#include "NSceneConfig.hpp"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    bool testgeo(false); 
    BOpticksResource okr(testgeo) ;  // no Opticks at this level 

    const char* dbgmesh = SSys::getenvvar("DBGMESH");
    int dbgnode = SSys::getenvint("DBGNODE", -1) ; 

    const char* gltfbase = argc > 1 ? argv[1] : okr.getDebuggingIDFOLD() ;

    const char* gltfname = "g4_00.gltf" ;
    const char* gltfconfig = "check_surf_containment=0,check_aabb_containment=0" ; 

    LOG(info) << argv[0]
              << " gltfbase " << gltfbase
              << " gltfname " << gltfname
              << " gltfconfig " << gltfconfig
              ;


    if(!NGLTF::Exists(gltfbase, gltfname))
    {
        LOG(warning) << "no such scene at"
                     << " base " << gltfbase
                     << " name " << gltfname
                     ;
        return 0 ; 
    } 

    const char* idfold = NULL ;

    NSceneConfig* config = new NSceneConfig(gltfconfig);
    NScene* scene = NScene::Load( gltfbase, gltfname, idfold, config, dbgnode );
    assert(scene);

    scene->dumpCSG(dbgmesh); 


    return 0 ; 
}
