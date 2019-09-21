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

#include "SSys.hh"

#include "NGLM.hpp"
#include "NPY.hpp"
#include "DummyPhotonsNPY.hpp"

#include "Opticks.hh"
#include "OpticksPhoton.h"
#include "OContext.hh"
#include "OBuf.hh"
#include "TBuf.hh"
#include "OXPPNS.hh"

#include "OPTICKS_LOG.hh"

/**

compactionTest
=================

Usage::

    compactionTest 
    compactionTest --generateoverride -10  ## up the photons to 10M


Objective: download part of a GPU photon buffer (N,4,4) ie N*4*float4 
with minimal hostside memory allocation.

Thrust based approach:

* determine number of photons passing some criteria (eg with an associated PMT identifier)
* allocate temporary GPU hit_buffer and use thrust::copy_if to fill it  
* allocate host side hit_buffer sized appropriately and pull back the hits into it 

See thrustrap-/tests/TBuf4x4Test.cu for development of the machinery 
based on early demo code from env-;optixthrust- with the simplification 
of using a float4x4 type for the Thrust photons buffer description.

**/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv, "--compute");  // --printenabled
    ok.configure() ;
 
    unsigned generateoverride = ok.getGenerateOverride() ;  // --generateoverride
    unsigned num_photons = generateoverride == 0 ? 5000 : generateoverride  ; 
    unsigned modulo = 1000 ;   
    bool integral_multiple = num_photons % modulo == 0 ; 
    unsigned x_num_hits = num_photons / modulo ; 
    bool verbose = num_photons <= 5000 ; 

    LOG(info)
        << " generateoverride " << generateoverride
        << " num_photons " << num_photons
        << " modulo " << modulo 
        << " integral_multiple " << integral_multiple
        << " x_num_hits " << x_num_hits
        << " verbose " << verbose 
        ;  

    assert( integral_multiple && "num_photons must be an an integral multiple of modulo " ); 


    unsigned PNUMQUAD = 4 ;
    unsigned hitmask = SURFACE_DETECT ; 
    LOG(error) << " hitmask " << hitmask ;  

    LOG(error) << "[ cpu generate " ; 
    NPY<float>* pho = DummyPhotonsNPY::Make(num_photons, hitmask, modulo );
    LOG(error) << "] cpu generate " ; 

    if(verbose)
    { 
        pho->save("$TMP/okop/compactionTestDummyPhotons.npy"); 
    }


    OContext* ctx = OContext::Create(&ok); 
    optix::Context context = ctx->getContext(); 

    int entry = ctx->addEntry("compactionTest.cu", "compactionTest", "exception");

    optix::Buffer photon_buffer = context->createBuffer( RT_BUFFER_INPUT );
    photon_buffer->setFormat(RT_FORMAT_FLOAT4);
    photon_buffer->setSize(num_photons*PNUMQUAD) ; 

    OBuf* pbuf = new OBuf("photon",photon_buffer);

    // PNUMQUAD formerly set here 
    context["photon_buffer"]->setBuffer(photon_buffer);  
    context["compaction_param"]->setUint(optix::make_uint2(PNUMQUAD, 0));

    LOG(error) << "[ prelaunch " ; 
    ctx->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, NULL);
    LOG(error) << "] prelaunch " ; 

    LOG(error) << "[ upload " ; 
    OContext::upload<float>( photon_buffer, pho );
    LOG(error) << "] upload " ; 

    LOG(error) << "[ launch " ; 
    ctx->launch( OContext::LAUNCH, entry, num_photons , 1, NULL ); 
    LOG(error) << "] launch " ; 


    CBufSpec cpho = pbuf->bufspec();   // getDevicePointer happens here with OBufBase::bufspec
    bool match = cpho.size == 4*num_photons ;
 
    if(!match)
        LOG(fatal) << " MISMATCH " 
                   << " cpho.size " <<  cpho.size
                   << " 4*num_photons " <<  4*num_photons
                   ;
    assert(match);  

    cpho.size = num_photons ;   //  decrease size by factor of 4 in order to increase "item" from 1*float4 to 4*float4 

    cpho.Summary("CBufSpec.Summary.cpho before TBuf"); 

    TBuf tpho("tpho", cpho );

    assert( tpho.getSize() == cpho.size ) ; 

    LOG(error) << " created tpho " 
               << " cpho.size : " << cpho.size 
               << " num_photons : " << num_photons 
               ; 

    //tpho.dump<unsigned>("tpho.dump<unsigned>(16,4*3+0,16*num_photons)", 16, 4*3+0, 16*num_photons );
    //tpho.dump<unsigned>("tpho.dump<unsigned>(16,4*3+3,16*num_photons)", 16, 4*3+3, 16*num_photons );

    NPY<float>* hit = NPY<float>::make(0,4,4);
    
    unsigned mskhis = hitmask ; 
    LOG(error) << "[ tpho.downloadSelection4x4 "; 
    tpho.downloadSelection4x4("thit<float4x4>", hit, mskhis, verbose );
    LOG(error) << "] tpho.downloadSelection4x4 "; 
    
    if(verbose)
    {
        const char* path = "$TMP/okop/compactionTestHits.npy";
        hit->save(path);
        SSys::npdump(path, "np.int32");
        SSys::npdump(path, "np.float32");
    }

    unsigned num_hits = hit->getNumItems(); 
    LOG(info) 
         << " num_hits " << num_hits  
         << " x_num_hits " << x_num_hits  
         ;

    assert( num_hits == x_num_hits );  

    delete ctx ; 

    return 0 ; 
}


