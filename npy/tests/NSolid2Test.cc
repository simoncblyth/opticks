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

// TEST=NSolid2Test om-t

#include "OPTICKS_LOG.hh"
#include "NSolid.hpp"
#include "NNode.hpp"


void test_is_ellipsoid()
{
    LOG(info); 

    nnode* a = NSolid::createEllipsoid( "a", 1.f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( a->is_ellipsoid() == false ); 
    
    nnode* b = NSolid::createEllipsoid( "b", 2.f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( b->is_ellipsoid() == true ); 

    nnode* c = NSolid::createEllipsoid( "c", 1.01f, 1.f, 1.f,  -1.f, 1.f  ) ; 
    assert( c->is_ellipsoid() == true ); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_is_ellipsoid();
 
    return 0 ; 
}
