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

optixtest-ver(){ echo OptiX_380 ; }
optixtest-inc(){ echo /Developer/$(optixtest-ver)/include ; }
optixtest-lib(){ echo /Developer/$(optixtest-ver)/lib64; }
  
optixtest-nvcc()
{
    local cu=$1 
    local ptx=$2
    local inc=$(optixtest-inc)

    #nvcc -arch=sm_30 -m64 -std=c++11 -O2 -use_fast_math -ptx $cu -I$inc -o $ptx
    #nvcc -arch=sm_30 -m64 -std=c++11  -use_fast_math -ptx $cu -I$inc -o $ptx
    nvcc -arch=sm_30 -m64 -std=c++11   -ptx $cu -I$inc -o $ptx
}

optixtest()
{
    # expects to be invoked from optixrap/cu 
    # and to find nam.cu ../tests/nam.cc

    local nam=${1:-cbrtTest}
    local fun=${nam}Callable

    local exe=/tmp/$nam
    local ptx=/tmp/$nam.ptx
    local ptxf=/tmp/$fun.ptx

    local cc=../tests/$nam.cc
    local cu=$nam.cu
    local cuf=$fun.cu

    local inc=$(optixtest-inc)
    local lib=$(optixtest-lib)

    clang -std=c++11 -I/usr/local/cuda/include -I$inc -L$lib -loptix  -lc++  -Wl,-rpath,$lib  $cc  -o $exe

    optixtest-nvcc $nam.cu $ptx

    if [ -f "$cuf" ]; then

        optixtest-nvcc $cuf $ptxf

        echo $exe $ptx $nam exception $ptxf $fun
        $exe $ptx $nam exception $ptxf $fun

    else
        echo $exe $ptx $nam

        #export OPTIX_API_CAPTURE=1

        $exe $ptx $nam
        #unset OPTIX_API_CAPTURE
    fi
}
optixtest
optixtest cbrtTest 

*/

/*
 NB this is for minimal testing ... so only standard and optix headers are allowed
*/


#include <optixu/optixpp_namespace.h>
#include <string>

struct OptiXMinimalTest 
{
   int    m_argc ; 
   char** m_argv ; 

   const char* m_ptxpath ; 
   const char* m_raygen_name ; 
   const char* m_exception_name ; 
   const char* m_callable_ptxpath ; 
   const char* m_callable_name ; 
     
   OptiXMinimalTest(optix::Context& context, int argc, char** argv ); 
   std::string description();

   void init(optix::Context& context);  
};


#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>

OptiXMinimalTest::OptiXMinimalTest(optix::Context& context, int argc, char** argv )
    :
    m_argc(argc),
    m_argv(argv),
    m_ptxpath(         argc > 1 ? argv[1] : NULL),
    m_raygen_name(     argc > 2 ? argv[2] : NULL),
    m_exception_name(  argc > 3 ? argv[3] : NULL),
    m_callable_ptxpath(argc > 4 ? argv[4] : NULL),
    m_callable_name(   argc > 5 ? argv[5] : NULL)
{
    init(context);
}

void OptiXMinimalTest::init(optix::Context& context)
{
    context->setEntryPointCount( 1 );
    context->setPrintEnabled(true);
    context->setPrintBufferSize(2*2*2*8192);

    optix::Program raygenProg    = context->createProgramFromPTXFile(m_ptxpath, m_raygen_name);
    optix::Program exceptionProg = context->createProgramFromPTXFile(m_ptxpath, m_exception_name);

    context->setRayGenerationProgram(0,raygenProg);
    context->setExceptionProgram(0,exceptionProg);

    if(!m_callable_ptxpath) return ; 

    optix::Buffer callable = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1 );
    int* callable_id = static_cast<int*>( callable->map() );

    std::cout << " callable_ptxpath " << m_callable_ptxpath
              << " callable_name "    << m_callable_name 
              << std::endl ;  

    optix::Program callableProg = context->createProgramFromPTXFile( m_callable_ptxpath, m_callable_name ); 
    callable_id[0] = callableProg->getId();
    callable->unmap();

    raygenProg["callable"]->set( callable );

}

std::string OptiXMinimalTest::description()
{
    std::stringstream ss ; 
    ss  
              << " ptxpath " << m_ptxpath
              << " raygen " << m_raygen_name 
              << " exception " << ( m_exception_name ? m_exception_name : "-" )
              << " callable_ptxpath " << ( m_callable_ptxpath ?  m_callable_ptxpath : "-" )
              << " callable_name " << ( m_callable_name ? m_callable_name  : "-" )
              ;

    return ss.str(); 
}


