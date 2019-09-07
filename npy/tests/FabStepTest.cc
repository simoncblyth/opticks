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

#include "FabStepNPY.hpp"

#include "SSys.hh"
#include "NPY.hpp"

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"


unsigned TORCH     =  0x1 << 12 ; 
unsigned FABRICATED = 0x1 << 15 ; 


void test_fabstep_0()
{
    unsigned nstep = 10 ; 
    FabStepNPY* fab = new FabStepNPY(FABRICATED, nstep, 100 ) ;  
    NPY<float>* npy = fab->getNPY();

    const char* path = "$TMP/fabstep_0.npy" ;
    npy->save(path);

    SSys::npdump(path, "np.int32");
}

int main(int argc, char** argv )
{

    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 

    test_fabstep_0();


    return 0 ;
}


