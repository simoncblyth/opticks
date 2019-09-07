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

/*
   For debugging torch configuration..
   this test has to live in ggeo- rather than npy- 
   as ggeo- info is required for targetting


   ggv --torchstep  "frame=3201;source=0,0,1000;target=0,0,0;radius=300;"   "frame=3153;source=0,0,1000;target=0,0,0;radius=300;" 
*/


#include "SSys.hh"

#include "NGLM.hpp"  // npy-
#include "NPY.hpp"

#include "Opticks.hh"     // okc-
#include "OpticksHub.hh"  // okg-
#include "OpticksGen.hh"

#include "TorchStepNPY.hpp" //npy-

#include "OKGEO_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKGEO_LOG__ ; 


    Opticks ok(argc, argv);

    OpticksHub hub(&ok);

    OpticksGen* gen = hub->getGen();

    TorchStepNPY* ts = gen->makeTorchstep() ;

    ts->dump(argv[0]);

    NPY<float>* gs = ts->getNPY();

    const char* path = "$TMP/TorchStepNPYTest.npy" ;

    gs->save(path);

    SSys::npdump(path, "np.int32"); 

    return 0 ;
}


