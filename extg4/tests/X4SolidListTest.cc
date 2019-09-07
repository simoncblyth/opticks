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

#include <cassert>
#include "G4Sphere.hh"

#include "X4Solid.hh"
#include "X4SolidList.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);
    
    G4Sphere* s1 = X4Solid::MakeSphere("s1", 100.f, 0.f); 
    G4Sphere* s2 = X4Solid::MakeSphere("s2", 100.f, 0.f); 

    LOG(info) << "s1:\n" << *s1 ; 
    LOG(info) << "s2:\n" << *s2 ; 


    X4SolidList sl ; 

    sl.addSolid(s1); 
    sl.addSolid(s1); 
    sl.addSolid(s1); 

    sl.addSolid(s2); 

    assert( sl.getNumSolids() == 2 ); 

    LOG(info) << sl.desc() ; 
 
    return 0 ; 
}
