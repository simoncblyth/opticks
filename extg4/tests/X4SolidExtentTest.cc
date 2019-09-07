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

#include "X4SolidExtent.hh"
#include "G4VSolid.hh"
#include "G4Orb.hh"

#include "NBBox.hpp"
#include "NGLM.hpp"

#include "OPTICKS_LOG.hh"

void test_Extent()
{
    G4Orb* orb = new G4Orb("orb", 100); 
    nbbox* bb = X4SolidExtent::Extent(orb); 
    bb->dump();
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_Extent(); 
    
    return 0 ; 
}


