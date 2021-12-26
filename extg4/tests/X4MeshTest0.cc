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

#include "G4Sphere.hh"
#include "X4Solid.hh"
#include "X4Mesh.hh"

#include "OPTICKS_LOG.hh"
#include "X4_GetSolid.hh"

void test_convert_save()
{
    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere", 100.f, 0.f); 

    std::cout << *sp << std::endl ; 

    X4Mesh* xm = new X4Mesh(sp, -1) ; 

    LOG(info) << xm->desc() ; 

    xm->save("/tmp/X4MeshTest/X4MeshTest.gltf"); 
}


void test_placeholder()
{
    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere", 100.f, 0.f); 

    GMesh* pl = X4Mesh::Placeholder(sp );
 
    assert( pl ); 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_convert_save(); 
    //test_placeholder();

    return 0 ; 
}
