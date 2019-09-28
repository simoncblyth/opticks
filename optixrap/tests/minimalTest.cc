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

#include "OptiXTest.hh"
#include "OContext.hh"
#include "NPY.hpp"
#include "OPTICKS_LOG.hh"

const char* TMPDIR = "$TMP/optixrap/minimalTest" ; 

int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    OContext::SetupOptiXCachePathEnvvar(); 
    optix::Context context = optix::Context::create();

    //const char* buildrel = "optixrap/tests" ; 
    const char* ptxrel = "tests" ; 
    const char* cmake_target = "minimalTest" ; 
    OptiXTest* test = new OptiXTest(context, "minimalTest.cu", "minimal", "exception", ptxrel, cmake_target ) ;
    test->Summary(argv[0]);

    // for inknown reasons this has become slow, taking 30s for 512x512
    //unsigned width = 512 ; 
    //unsigned height = 512 ; 

    unsigned width = 16 ; 
    unsigned height = 16 ; 



    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);

    context->validate();
    context->compile();

    context->launch(0, width, height);


    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

    const char* name = "minimalTest.npy";
    npy->save(TMPDIR, name);

    return 0;
}
