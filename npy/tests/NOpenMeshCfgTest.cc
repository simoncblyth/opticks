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

#ifdef OLD_PARAMETERS
#include "X_BParameters.hh"
#else
#include "NMeta.hpp"
#endif

#include "NOpenMeshCfg.hpp"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

#ifdef OLD_PARAMETERS
    X_BParameters meta ; 
#else
    NMeta meta ; 
#endif

    meta.add<std::string>("poly", "BSP");
    meta.add<std::string>("polycfg", "contiguous=1,reversed=0,numsubdiv=3,offsave=1");

    const char* treedir = NULL ; 

    NOpenMeshCfg cfg(&meta, treedir)  ;
    LOG(info) << cfg.desc() ;  

    return 0 ; 
}
