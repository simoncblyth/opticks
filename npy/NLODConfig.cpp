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

#include "BConfig.hh"
#include "PLOG.hh"
#include "NLODConfig.hpp"


const char* NLODConfig::instanced_lodify_onload_ = ">0 : Apply LODification (GMergedMesh::MakeLODComposite) on loading non-global GMergedMesh in GGeoLib " ;  


NLODConfig::NLODConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    levels(3),
    instanced_lodify_onload(0)
{
    LOG(verbose) << "NLODConfig::NLODConfig"
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    // TODO: incorp the help strings into the machinery and include in dumping 

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("levels", &levels );
    bconfig->addInt("instanced_lodify_onload", &instanced_lodify_onload );

    bconfig->parse();
}

void NLODConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

