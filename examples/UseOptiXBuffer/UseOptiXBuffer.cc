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

#include <array>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>
#include <optix.h>
#include <iostream>
#include "OKConf.hh"

//#define WITH_NPY 1
#ifdef WITH_NPY
#include "NPY.hpp"
#endif

// from SDK/sutil/sutil.h
struct APIError
{   
    APIError( RTresult c, const std::string& f, int l ) 
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Error check/report helper for users of the C API 
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)




int main()
{
    const int width = 10u ; 
#ifdef WITH_NPY
    NPY<float>* arr = NPY<float>::make( width ); 
    arr->zero();
#else
    std::array<float, width> arr ;  
#endif

    const char* cmake_target = "UseOptiXBuffer" ;
    const char* cu_name = "UseOptiXBuffer.cu" ; 
    const char* ptxpath = OKConf::PTXPath( cmake_target, cu_name ); 
    const char* progname = "readOnly" ;
    std::cout << "ptxpath : [" << ptxpath << "]" << std::endl ; 


    RTcontext context = 0;
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );

    RT_CHECK_ERROR( rtContextSetPrintEnabled( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetPrintBufferSize( context, 4096 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) ); 

    RTbuffer  buffer;
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT ) );
    RT_CHECK_ERROR( rtBufferSetSize1D( buffer, width ) );

    RTvariable result_buffer ; 
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer", &result_buffer ) );
    RT_CHECK_ERROR( rtVariableSetObject( result_buffer, buffer ) );

    RTprogram raygen ;
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context, ptxpath, progname, &raygen )) ;

    unsigned entry_point_index = 0u ;  
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, entry_point_index, raygen )); 
    RT_CHECK_ERROR( rtContextValidate( context ) );

    RTsize w = width ; 
    RT_CHECK_ERROR( rtContextLaunch1D( context, entry_point_index, w ) );

    void* data ; 
    RT_CHECK_ERROR( rtBufferMap( buffer, &data ) );
#ifdef WITH_NPY
    arr->read(data); 
#else
    memcpy( arr.data(), data, sizeof(float)*width ) ;
#endif
    RT_CHECK_ERROR( rtBufferUnmap( buffer )) ;

#ifdef WITH_NPY
    arr->dump();
#else
    for(int i=0 ; i < width ; i++ ) 
    {
        std::cout << arr[i] << " " ; 
        assert( arr[i] == 42.f );  
    }
    std::cout << std::endl ; 
#endif

    return 0 ; 
}

