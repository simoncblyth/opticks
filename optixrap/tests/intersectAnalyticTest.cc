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

/**
::

    intersectAnalyticTest --cu dummyTest.cu
    intersectAnalyticTest --cu torusTest.cu            
    intersectAnalyticTest --cu sphereTest.cu
    intersectAnalyticTest --cu coneTest.cu
    intersectAnalyticTest --cu convexpolyhedronTest.cu

**/
#include "OptiXTest.hh"

#include "SPath.hh"
#include "OGeo.hh"
#include "NPY.hpp"
#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);
    const SAr& args = PLOG::instance->args ; 
    args.dump(); 

    const char* cu_name = args.get_arg_after("--cu", NULL ); 

    if(cu_name == NULL)
    {
        LOG(fatal) << " require \"--cu name.cu\" argument " ; 
        return 0 ; 
    } 

    const char* progname = SPath::Stem(cu_name) ;        

    LOG(info) 
         << " cu_name " << cu_name 
         << " progname " << progname 
         ;

    optix::Context context = optix::Context::create();

    RTsize stack_size = context->getStackSize(); 
    LOG(info) << " stack_size " << stack_size ; 
    //context->setStackSize(6000);


    const char* buildrel = "optixrap/tests" ; 
    const char* cmake_target = "intersectAnalyticTest" ;  

    OptiXTest* test = new OptiXTest(context, cu_name, progname, "exception", buildrel, cmake_target ) ;

    std::cout << test->description() << std::endl ; 

    unsigned width = 1 ; 
    unsigned height = 1 ; 

    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);


    NPY<float>* planBuf = NPY<float>::make(6, 4) ;  
    planBuf->zero();
    float hsize = 200.f ;
    unsigned j = 0 ; 
 
    planBuf->setQuad(0,j,  1.f, 0.f, 0.f,hsize );
    planBuf->setQuad(1,j, -1.f, 0.f, 0.f,hsize );
    planBuf->setQuad(2,j,  0.f, 1.f, 0.f,hsize );
    planBuf->setQuad(3,j,  0.f,-1.f, 0.f,hsize );
    planBuf->setQuad(4,j,  0.f, 0.f, 1.f,hsize );
    planBuf->setQuad(5,j,  0.f, 0.f,-1.f,hsize );

    unsigned verbosity = 3 ; 

    const char* ctxname = progname ;  // just informational

    optix::Buffer planBuffer = OGeo::CreateInputUserBuffer<float>( context, planBuf,  4*4, "planBuffer", ctxname, verbosity); 
    context["planBuffer"]->setBuffer(planBuffer);

    context->validate();
    context->compile();
    context->launch(0, width, height);


    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

    const char* path = "$TMP/oxrap/intersectAnalyticTest.npy";
    std::cerr << "save result npy to " << path << std::endl ; 
 
    npy->save(path);



    return 0;
}
