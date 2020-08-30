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
#include "UseOptiXTextureLayered.h"


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


/**
optix7-;optix6-p 21

As of version 3.9, OptiX supports cube, layered, and mipmapped textures using
new API calls rtBufferMapEx, rtBufferUnmapEx, rtBufferSetMipLevelCount.1
Layered textures are equivalent to CUDA layered textures and OpenGL texture
arrays. They are created by calling rtBufferCreate with RT_BUFFER_LAYERED and
cube maps by passing RT_BUFFER_CUBEMAP. In both cases the buffer’s depth
dimension is used to specify the number of layers or cube faces, not the depth
of a 3D buffer.  OptiX programs can access texture data with CUDA C’s built-in
tex1D, tex2D and tex3D functions.

**/



int main()
{
    const int nx = 4u ; 
    const int ny = 4u ; 
    const int nz = 4u ;
    const int size = nx*ny*nz ; 

    float* values = NULL ; 
 
#ifdef WITH_NPY
    std::cout << " WITH NPY " << std::endl ; 
    NPY<float>* inp = NPY<float>::make( size ); 
    inp->fill(42.f);
    inp->dump();
    values = inp->getValues();
    NPY<float>* out = NPY<float>::make( size ); 
    out->zero();
#else
    std::cout << " NOT WITH NPY " << std::endl ; 
    std::array<float, size> inp ;  
    inp.fill(42.f);
    values = inp.data();
    std::array<float, size> out ;  
    out.fill(0.f);  
#endif

    // makes more sense for the layer to be first index 

    for(int i=0 ; i < nx ; i++){
    for(int j=0 ; j < ny ; j++){
    for(int k=0 ; k < nz ; k++){   

       int index = i*ny*nz + j*nz + k ; 
       *(values + index) = float(index) ;

    }   
    }   
    }   

    const char* cmake_target = "UseOptiXTextureLayered" ;
    const char* cu_name = "UseOptiXTextureLayered.cu" ; 
    const char* ptxpath = OKConf::PTXPath( cmake_target, cu_name ); 
    const char* progname = "readWrite" ;
    std::cout 
        << " ptxpath: [" << ptxpath << "]" 
        << " progname: [" << progname << "]" 
#ifdef FROM_BUF
        << " FROM_BUF is enabled "
#else
        << " FROM_BUF is NOT enabled "
#endif
#ifdef FROM_NPY
        << " FROM_NPY is enabled "
#else
        << " FROM_NPY is NOT enabled "
#endif
        << std::endl
        ; 

    RTcontext context = 0;
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );

    RT_CHECK_ERROR( rtContextSetPrintEnabled( context, 1 ) );
    RT_CHECK_ERROR( rtContextSetPrintBufferSize( context, 4096 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) ); 

    RTbuffer tex_buffer;
    unsigned bufferdesc = RT_BUFFER_INPUT | RT_BUFFER_LAYERED ; 
    //  If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.

    RT_CHECK_ERROR( rtBufferCreate( context, bufferdesc, &tex_buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( tex_buffer, RT_FORMAT_FLOAT ) );
    RT_CHECK_ERROR( rtBufferSetSize3D( tex_buffer, nx, ny, nz ) );
    unsigned levels = 1 ; 
    RT_CHECK_ERROR( rtBufferSetMipLevelCount( tex_buffer, levels ) );

    RTvariable tex_buffer_variable ; 
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "tex_buffer", &tex_buffer_variable ) );
    RT_CHECK_ERROR( rtVariableSetObject( tex_buffer_variable, tex_buffer ) );

    // listing 3.23  optix6-p 20

    std::cout << "[ creating tex_sampler " << std::endl ;  
    RTtexturesampler tex_sampler;
    RT_CHECK_ERROR( rtTextureSamplerCreate( context, &tex_sampler ) );

    //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 

    RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 0, wrapmode ) );
    RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 1, wrapmode ) );
    RT_CHECK_ERROR( rtTextureSamplerSetWrapMode( tex_sampler, 2, wrapmode ) );

    RTfiltermode minmag = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = minmag ; 
    RTfiltermode magnification = minmag ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ; 

    RT_CHECK_ERROR( rtTextureSamplerSetFilteringModes( tex_sampler, minification, magnification, mipmapping ) );

    // indexmode : controls the interpretation of texture coordinates
    //RTtextureindexmode indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ;  // parametrized over [0,1]
    RTtextureindexmode indexmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // texture coordinates are interpreted as array indices into the contents of the underlying buffer object
    RT_CHECK_ERROR( rtTextureSamplerSetIndexingMode( tex_sampler, indexmode ) );

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ; // return floating point values normalized by the range of the underlying type
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;  // return data of the type of the underlying buffer
    // when the underlying type is float the is no difference between RT_TEXTURE_READ_NORMALIZED_FLOAT and RT_TEXTURE_READ_ELEMENT_TYPE

    RT_CHECK_ERROR( rtTextureSamplerSetReadMode( tex_sampler, readmode ) );
    RT_CHECK_ERROR( rtTextureSamplerSetMaxAnisotropy( tex_sampler, 1.0f ) );
    std::cout << "] creating tex_sampler " << std::endl ;  

#ifdef FROM_BUF
    std::cout << ". FROM_BUF skipping sampler hookup " << std::endl ;  
#else
    std::cout << "[ associate tex_sampler with tex_buffer " << std::endl ;  
    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    RT_CHECK_ERROR( rtTextureSamplerSetBuffer( tex_sampler, deprecated0, deprecated1, tex_buffer ) );
    std::cout << "] associate tex_sampler with tex_buffer " << std::endl ;  


    std::cout << "[ get texture_id of tex_sampler " << std::endl ;  
    int texture_id(-1) ; 
    RT_CHECK_ERROR( rtTextureSamplerGetId( tex_sampler, &texture_id ) ); 
    std::cout << "] get texture_id of tex_sampler " << texture_id << std::endl ;  


    std::cout << "[ creating tex_sampler_variable " << std::endl ;  
    RTvariable tex_sampler_variable ; 
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "tex_sampler", &tex_sampler_variable ) );
    RT_CHECK_ERROR( rtVariableSetObject( tex_sampler_variable, tex_sampler ) );
    std::cout << "] creating tex_sampler_variable " << std::endl ;  

    RTvariable tex_param ;
    std::cout << "[ creating tex_param_variable " << std::endl ;  
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "tex_param", &tex_param ) );
    RT_CHECK_ERROR( rtVariableSet4i( tex_param, texture_id, 214, 42, 0 ) );
    std::cout << "] creating tex_param_variable " << std::endl ;  

#endif

    std::cout << "[ creating out_buffer " << std::endl ;  
    RTbuffer  out_buffer;
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &out_buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( out_buffer, RT_FORMAT_FLOAT ) );
    RT_CHECK_ERROR( rtBufferSetSize3D( out_buffer, nx, ny, nz ) );
    std::cout << "] creating out_buffer " << std::endl ;  

    std::cout << "[ creating out_buffer_variable " << std::endl ;  
    RTvariable out_buffer_variable ; 
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "out_buffer", &out_buffer_variable ) );
    RT_CHECK_ERROR( rtVariableSetObject( out_buffer_variable, out_buffer ) );
    std::cout << "] creating out_buffer_variable " << std::endl ;  

    std::cout << "[ creating raygen " << std::endl ;  
    RTprogram raygen ;
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context, ptxpath, progname, &raygen )) ;
    std::cout << "] creating raygen " << std::endl ;  

    std::cout << "[ setting raygen into context " << std::endl ;  
    unsigned entry_point_index = 0u ;  
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, entry_point_index, raygen )); 
    std::cout << "] setting raygen into context " << std::endl ;  

    std::cout << "[ contextValidate " << std::endl ;  
    RT_CHECK_ERROR( rtContextValidate( context ) );
    std::cout << "] contextValidate " << std::endl ;  


    bool exfill = false ;  
    //bool exfill = true ;  // terminates with APIError on both OptiX 5 and 6 
    std::cout << "[ uploading to tex buffer exfill  " << exfill << std::endl ;  
    if(exfill == false)
    {
        void* tex_data ; 
        RT_CHECK_ERROR( rtBufferMap( tex_buffer, &tex_data ) );
        #ifdef WITH_NPY
        inp->write(tex_data); 
        #else
        memcpy( tex_data, inp.data(), sizeof(float)*size ) ;
        #endif
        RT_CHECK_ERROR( rtBufferUnmap( tex_buffer )) ;
    }
    else
    {    
        for(int i=0 ; i < nx ; i++)
        {
            void* tex_data ; 
            //unsigned map_flags = RT_BUFFER_MAP_READ ; 
            unsigned map_flags = RT_BUFFER_MAP_READ_WRITE ; 
            unsigned layer = i ; 
            void* user_owned_must_be_null = NULL ; 

            RT_CHECK_ERROR( rtBufferMapEx( tex_buffer, map_flags, layer, user_owned_must_be_null,  &tex_data ) );
            #ifdef WITH_NPY
            inp->writeItem(tex_data, layer);   // NB first array index  
            #else
            memcpy( tex_data, inp.data() + layer*ny*nz , sizeof(float)*ny*nz ) ;
            #endif
            RT_CHECK_ERROR( rtBufferUnmapEx( tex_buffer, layer )) ;
        }
    }
    std::cout << "] uploading to tex buffer exfill  " << exfill << std::endl ;  



    std::cout << "[ launch " << std::endl ;  
    RT_CHECK_ERROR( rtContextLaunch3D( context, entry_point_index, nx, ny, nz ) );
    std::cout << "] launch " << std::endl ;  

    void* out_data ; 
    RT_CHECK_ERROR( rtBufferMap( out_buffer, &out_data ) );
#ifdef WITH_NPY
    out->read(out_data); 
#else
    memcpy( out.data(), out_data, sizeof(float)*size ) ;
#endif
    RT_CHECK_ERROR( rtBufferUnmap( out_buffer )) ;

#ifdef WITH_NPY
    out->dump();
#else
    int count=0 ; 
    for(int i=0 ; i < nx ; i++){
    for(int j=0 ; j < ny ; j++){
    for(int k=0 ; k < nz ; k++){   

       int index = i*ny*nz + j*nz + k ; 
       assert( count == index ); 
       count++ ; 

       float val = out[index] ; 
       std::cout << "(" << i << " " << j << " " << k << ") " << val << std::endl ;  

       int ival = int(val); 
       assert( ival == index );  

    }   
    }   
    }   
#endif
    return 0 ; 
}

