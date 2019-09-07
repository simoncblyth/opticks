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

// TEST=NNodeDumpTest om-t 

#include "NNode.hpp"
#include "NNodeSample.hpp"

#include "OPTICKS_LOG.hh"

typedef std::vector<nnode*> VN ;


void test_dump(const VN& nodes, unsigned idx)
{
    assert( idx < nodes.size() ) ; 
    nnode* n = nodes[idx] ; 
    LOG(info) << "\n" << " sample idx : " << idx ; 
    n->dump(); 
}

void test_dump(const VN& nodes)
{
    for(unsigned i=0 ; i < nodes.size() ; i++) test_dump(nodes, i) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    int idx = argc > 1 ? atoi(argv[1]) : -1 ; 

    if( idx == -1 ) 
    {
        test_dump(nodes); 
    }
    else
    {
        test_dump(nodes, idx); 
    }

    return 0 ; 
}



