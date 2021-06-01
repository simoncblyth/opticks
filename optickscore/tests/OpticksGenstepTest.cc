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

// TEST=OpticksGenstepTest om-t

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "NPY.hpp"
#include "TorchStepNPY.hpp"
#include "OpticksGenstep.hh"


const plog::Severity LEVEL = info ;  


void test_Dump()
{
    LOG(info) << "OpticksGenstep::Dump()" ;
    LOG(info) << std::endl << OpticksGenstep::Dump() ;
}

void test_load_and_dump()
{
    const SAr& args = PLOG::instance->args ;   
    const char* def = "$DATADIR/gensteps/dayabay/natural/1.npy" ;  // DATADIR is an internal "envvar"
    const char* path = args._argc > 1 ? args._argv[1] : def ; 

    BOpticksResource* rsc = BOpticksResource::Get(NULL) ;  // needed to resolve internal "envvar" DATADIR, see BResourceTest, BFile 
    assert(rsc); 

    LOG(info) << "path:" << path ; 
    NPY<float>* arr = NPY<float>::load(path) ; 
    if(arr == NULL) return ; 

    LOG(info) << "arr.shape:" << arr->getShapeString(); 
    OpticksGenstep* gs = new OpticksGenstep(arr) ; 

    unsigned modulo = 1000 ; 
    unsigned margin = 10 ;  
    gs->dump( modulo, margin ) ; 
}


void test_torchstep()
{
    unsigned gentype = OpticksGenstep_TORCH  ; 
    unsigned num_step = 1 ;  
    const char* config = NULL ;    

    assert( OpticksGenstep::IsTorchLike(gentype) ); 

    LOG(info) << " gentype " << gentype ; 

    TorchStepNPY* ts = new TorchStepNPY(gentype, config);

    glm::mat4 frame_transform(1.f);
    ts->setFrameTransform(frame_transform);

    for(unsigned i=0 ; i < num_step ; i++) 
    {    
        ts->addStep(); 
    }    

    NPY<float>* arr = ts->getNPY(); 

    arr->save("$TMP/debugging/OpticksGenstepTest/gs.npy");  

    const OpticksGenstep* gs = new OpticksGenstep(arr); 
    unsigned num_gensteps = gs->getNumGensteps(); 
    const NPY<float>* src = gs->getGensteps(); 

    LOG(LEVEL) 
        << " num_gensteps " << num_gensteps 
        << " src.shape " << src->getShapeString()
        ;   

    for(unsigned idx=0 ; idx < num_gensteps ; idx++)
    {   
        unsigned gentype = gs->getGencode(idx); 
        LOG(LEVEL) 
            << " idx " << idx 
            << " gentype " << gentype
            ;   

        if( OpticksGenstep::IsCerenkov(gentype) )
        {   
            assert(0); 
        }   
        else if( OpticksGenstep::IsScintillation(gentype) )
        {   
            assert(0); 
        }   
        else if( OpticksGenstep::IsTorchLike(gentype) )
        {   
            LOG(info) << " torch like " ; 
        }   
        else if( OpticksGenstep::IsMachinery(gentype) )
        {   
            assert(0); 
        }   
        else
        {   
            assert(0); 
        }   
    }   

    gs->dump("test_torchstep") ; 

}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_Dump(); 
    //test_load_and_dump(); 
    test_torchstep();

    return 0 ; 
}
