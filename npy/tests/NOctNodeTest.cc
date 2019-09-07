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

#include "NOctNode.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

   
    //int size = 1 << 5 ;   // 32
    int size = 1 << 4 ;   // 16

    nivec3 min(-size/2,-size/2,-size/2) ;

    float scale = 2.0f ;  // indice ints to real world floats 

    NOctNode* tree = NOctNode::Construct( min, size, scale); 

    NOctNode::Traverse(tree, 0);

    int num = NOctNode::TraverseIt(tree);

    LOG(info) << "NOctNode count " << num << " size*size*size " << size*size*size ;


    return 0 ; 
}


// hmm without simplification the Octress gains little
//  NOctNode count 1065 size*size*size 4096
//
