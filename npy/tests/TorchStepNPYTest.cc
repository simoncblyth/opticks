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
#include "SStr.hh"
#include "SSys.hh"

#include "NGLM.hpp"
#include "NPY.hpp"
#include "NStep.hpp"
#include "TorchStepNPY.hpp"


int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    unsigned gentype = 5  ;  // OpticksGenstep_TORCH  
    unsigned num_step = 3 ; 

    //const char* config = "target=3153;photons=10000;dir=0,1,0" ;
    const char* config = NULL ;

    LOG(info) << "[ts" ; 
    TorchStepNPY* ts = new TorchStepNPY(gentype, config);
    LOG(info) << "]ts" ; 

    // Normally frame targetting is done by okg/OpticksGen::targetGenstep 
    // with help of GGeo::getTransform but there is no geometry at this level : so set to identity. 

    int frameIdx = ts->getFrameIndex(); 
    assert( frameIdx == 0 ); 
     
    glm::mat4 identity(1.f); 
    LOG(info) << "[ts.setFrameTransform" ; 
    ts->setFrameTransform(identity);
    LOG(info) << "]ts.setFrameTransform" ; 

    for(unsigned i=0 ; i < num_step ; i++) 
    {
        LOG(info) << "[ts.addStep" ; 

        NStep* st = ts->getOneStep(); 

        st->setTime( 0.1*float(i+1) ); 
        ts->addStep();


        LOG(info) << "]ts.addStep" ; 
    }
 

    LOG(info) << "[ts.dump" ; 
    ts->dump();
    LOG(info) << "]ts.dump" ; 


    const char* path = "$TMP/torchstep.npy" ;
    NPY<float>* arr = ts->getNPY();
    arr->save(path);

    unsigned arr_items = arr->getNumItems() ; 
    LOG(info)
        << " arr " << arr->getShapeString()
        << " arr_items " << arr_items
        << " num_step " << num_step
        ;

    assert(arr_items == num_step);

    const char* cmd = SStr::Concat("python -c 'import os, numpy as np ; print(np.load(os.path.expandvars(\"", path, "\")).view(np.int32))' ");  
    system(cmd);
    SSys::npdump(path, "np.int32");
    SSys::npdump(path, "np.float32");


    return 0 ;
}

// om-;TEST=TorchStepNPYTest om-t



