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

// run this using optixbash bash function from OptiXMinimalTest.hh 
#include "OptiXMinimalTest.hh"


int main( int argc, char** argv ) 
{
    optix::Context context = optix::Context::create();

    RTsize stack_size_bytes = context->getStackSize() ;
    //stack_size_bytes *= 2 ; 
    //context->setStackSize(stack_size_bytes);
   
    OptiXMinimalTest* test = new OptiXMinimalTest(context, argc, argv  ) ;
    std::cout << test->description() << std::endl ; 

    //unsigned width = 512 ; 
    //unsigned height = 512 ; 

    unsigned width = 1 ; 
    unsigned height = 1 ; 

    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);

    context->validate();
    context->compile();
    context->launch(0, width, height);

    return 0;
}
