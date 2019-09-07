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

#include "TorchStepNPY.hpp"

#include "SSys.hh"
#include "NPY.hpp"

#ifdef _MSC_VER
// quell: object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"




int main(int argc, char** argv )
{

    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 


    TorchStepNPY* m_torchstep ; 


    unsigned TORCH = 4096 ; 
    unsigned nstep = 1 ; 
    //const char* config = "target=3153;photons=10000;dir=0,1,0" ;
    const char* config = NULL ;

    m_torchstep = new TorchStepNPY(TORCH, nstep, config);

    m_torchstep->dump();

    m_torchstep->addStep();

    NPY<float>* npy = m_torchstep->getNPY();
    npy->save("$TMP/torchstep.npy");

    assert(npy->getNumItems() == nstep);

    const char* cmd = "python -c 'import os, numpy as np ; print np.load(os.path.expandvars(\"$TMP/torchstep.npy\")).view(np.int32)' " ;
    system(cmd);

    SSys::npdump("$TMP/torchstep.npy", "np.int32");
    SSys::npdump("$TMP/torchstep.npy", "np.float32");


    return 0 ;
}


