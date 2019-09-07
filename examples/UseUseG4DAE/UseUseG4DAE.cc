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

#include <iostream>
#include "UseG4DAE.hh"

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage " << argv[0] << " /path/to/input.gdml /path/to/output.dae " << std::endl ; 
        return 1 ;  
    }

    const char* gdml_path = argv[1] ; 
    const char* dae_path = argv[2] ;
 
    UseG4DAE_gdml2dae( gdml_path, dae_path );    

    return 0 ; 
}


