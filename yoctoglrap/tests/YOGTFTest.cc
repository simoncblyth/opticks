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
#include "BStr.hh"
#include "BFile.hh"
#include "NGLM.hpp"

#include "YOG.hh"
#include "YOGTF.hh"

using YOG::Sc ; 
using YOG::Nd ; 

using YOG::TF ; 

const char* TMPDIR = "$TMP/yoctoglrap/YOGTFTest" ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Sc sc ; 

    LOG(info) << sc.desc() ; 

    int N = 3 ; 
    for(int i=0 ; i < N ; i++)
    {
        int ndIdx = sc.add_test_node(i);
        std::string lvName = "lvName" ; 
        std::string soName = "soName" ; 
        sc.add_mesh(i, lvName, soName );  

        assert( ndIdx == i ); 
        Nd* nd = sc.nodes.back() ;
        nd->children = { i+1, i+2 } ;  // purely dummy  

        std::cout << nd->desc() << std::endl ; 
    }    




    LOG(info) << sc.desc() ; 


    TF tf(&sc); 

    tf.save(TMPDIR, "YOGTFTest.gltf" );     


    return 0 ; 
}


