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

#include "G4Material.hh"

#include "X4Material.hh"
#include "X4OpNoviceMaterials.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4OpNoviceMaterials opnov ; 

    G4Material* water = opnov.water ;

    char mode_g4_interpolated_onto_domain = 'G' ;

    GMaterial* wine = X4Material::Convert(water, mode_g4_interpolated_onto_domain ) ; 

    wine->Summary();

    GMaterialLib::dump(wine) ; 

    return 0 ; 
}
