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


#include "OXPPNS.hh"
#include "OKConf.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    optix::Context context = optix::Context::create();
    context->setEntryPointCount(1);
    context->setRayTypeCount(1);    // <-- without this segments at launch (new behaviour in OptiX_600)  
    context->setPrintEnabled(true);

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;
    npy->zero();

    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT ) ;
    buffer->setFormat( RT_FORMAT_FLOAT4 );  
    buffer->setSize( ni*nj ) ; 

    context["output_buffer"]->set(buffer);


    const char* cmake_target = "writeBufferLowLevelTest" ; 
    const char* cu_name = "writeBufferLowLevelTest.cu" ;
    const char* ptxrel = "tests" ; 
    const char* ptx_path = OKConf::PTXPath(cmake_target, cu_name, ptxrel ); 
    const char* progname = "writeBuffer" ; 
    optix::Program program = context->createProgramFromPTXFile( ptx_path , progname ); 

    unsigned entry = 0 ; 
    context->setRayGenerationProgram( entry, program ); 

    unsigned width = ni ; 
    context->launch( entry, width  );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/writeBufferLowLevelTest.npy");

    return 0;
}
