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
#include "OpticksBufferControl.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv, "--compute");
    ok->configure();

    const char* cmake_target = "downloadTest" ;  
    const char* ptxrel = "tests" ; 

    OContext* ctx = OContext::Create(ok, cmake_target, ptxrel );
    optix::Context context = ctx->getContext(); 

    unsigned entry = ctx->addEntry("downloadTest.cu", "minimal", "exception");

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;
    npy->fill(42.);
    npy->save("$TMP/downloadTest_0.npy");


    //const char* ctrl = "OPTIX_SETSIZE,OPTIX_INPUT_OUTPUT" ;  <-- all zeros in OptiX 600 ?
    const char* ctrl = "OPTIX_OUTPUT_ONLY" ;    // <-- all zeros too ?

    npy->setBufferControl(OpticksBufferControl::Parse(ctrl));


    optix::Buffer buffer = ctx->createBuffer<float>( npy, "demo");
    context["output_buffer"]->set(buffer);

    ctx->launch( OContext::VALIDATE,  entry, ni, 1);
    ctx->launch( OContext::COMPILE,   entry, ni, 1);
    ctx->launch( OContext::PRELAUNCH, entry, ni, 1);
    ctx->launch( OContext::LAUNCH,    entry, ni, 1);

    npy->zero();

    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/downloadTest_1.npy");

    delete ctx ; 


    return 0;
}
