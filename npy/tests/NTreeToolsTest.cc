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

#include "NTreeTools.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


struct Node 
{  
   Node(int idx) : idx(idx) 
   {
       init();
   }
   void init(){ for(int i=0 ; i < 4 ; i++) children[i] = NULL ; }

   int idx ; 
   Node* children[4] ; 
};


template class NTraverser<Node,4> ; 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    Node root(0);
    root.children[0] = new Node(1) ; 
    root.children[3] = new Node(3) ; 

    NTraverser<Node,4> tv(&root, "NTraverser", 1 ) ; 


    return 0 ; 
}
