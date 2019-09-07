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

#include "OPTICKS_LOG.hh"
#include "Randomize.hh"
#include "CMixMaxRng.hh"


void dump_flat(int n)
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;

    long seed = engine->getSeed() ; 
    LOG(info) 
        << " seed " << seed 
        << " name " << engine->name() 
        ; 

    for(int i=0 ; i < n ; i++)
    {
        double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
        std::cout << u << std::endl ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    CMixMaxRng mmr ;
    dump_flat(10); 

    return 0 ; 
}


