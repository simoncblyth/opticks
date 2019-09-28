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
#include "NGLM.hpp"

#include "YOG.hh"

using YOG::Sc ; 
using YOG::Nd ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Sc sc ; 
    LOG(info) << sc.desc() ; 

    for(int i=0 ; i < 10 ; i++)
    {
        int ndIdx = sc.add_test_node(i);
        assert( ndIdx == i ); 
        Nd* ndback = sc.nodes.back() ;
        std::cout << ndback->desc() << std::endl ; 
    }    

    LOG(info) << sc.desc() ; 

    return 0 ; 
}


