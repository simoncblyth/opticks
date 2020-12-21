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

#include "Opticks.hh"
#include "GMaterialLib.hh"
#include "X4MaterialTable.hh"
#include "X4OpNoviceMaterials.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4OpNoviceMaterials opnov ; 

    assert( opnov.water && opnov.air ) ; 

    Opticks ok(argc, argv, "--allownokey");
    ok.configure();

    GMaterialLib* mlib = new GMaterialLib(&ok);

    std::vector<G4Material*> material_with_efficiency ; 

    X4MaterialTable::Convert(mlib, material_with_efficiency) ; 

    assert( mlib->getNumMaterials() == 2 ); 

    mlib->dump();

    return 0 ; 
}
