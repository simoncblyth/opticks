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

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include <iostream>
#include "OKConf.hh"
#include "UseOptiXTextureLayeredPP.h"

#ifdef WITH_NPY
#include "NPY.hpp"
#endif

/**
TODO:

1. tighten up, make it higher level by pulling off common stuff into OConfig for example
2. expt with realistic 2d theta-phi layered spherical texture : especially wrt the wrapping mode, passing domain range 
3. apply 2d theta-phi layered texture to some instanced geometry and shade based on it : eg make a variety of PPM beach-balls  

**/

int main()
{
    const int nx = 10u ; 
    const int ny = 10u ; 
    const int nz = 10u ;
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

    for(int i=0 ; i < nx ; i++){
    for(int j=0 ; j < ny ; j++){
    for(int k=0 ; k < nz ; k++){   

       int index = i*ny*nz + j*nz + k ; 
       *(values + index) = float(index) ;

    }   
    }   
    }   

    const char* cmake_target = "UseOptiXTextureLayeredPP" ;
    const char* cu_name = "UseOptiXTextureLayeredPP.cu" ; 
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

    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1); 
    context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
    context->setPrintEnabled(1);  
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);
    unsigned entry_point_index = 0u ; 



    unsigned bufferdesc = RT_BUFFER_INPUT | RT_BUFFER_LAYERED ; 
    //  If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 
    texBuffer->setFormat( RT_FORMAT_FLOAT ); 
    texBuffer->setSize(nx, ny, nz); 

    std::cout << "[ creating tex_sampler " << std::endl ;  
    optix::TextureSampler tex = context->createTextureSampler(); 

    //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 
    tex->setWrapMode(0, wrapmode);
    tex->setWrapMode(1, wrapmode);
    //tex->setWrapMode(2, wrapmode);   corresponds to layer?

    RTfiltermode minmag = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = minmag ; 
    RTfiltermode magnification = minmag ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ; 

    tex->setFilteringModes(minification, magnification, mipmapping);

    // indexmode : controls the interpretation of texture coordinates
    //RTtextureindexmode indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ;  // parametrized over [0,1]
    RTtextureindexmode indexmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // texture coordinates are interpreted as array indices into the contents of the underlying buffer object
    tex->setIndexingMode( indexmode );  

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ; // return floating point values normalized by the range of the underlying type
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;  // return data of the type of the underlying buffer
    // when the underlying type is float the is no difference between RT_TEXTURE_READ_NORMALIZED_FLOAT and RT_TEXTURE_READ_ELEMENT_TYPE

    tex->setReadMode( readmode ); 
    tex->setMaxAnisotropy(1.0f);
    std::cout << "] creating tex_sampler " << std::endl ;  

#ifdef FROM_BUF
    std::cout << ". FROM_BUF skipping sampler hookup " << std::endl ;  
#else
    std::cout << "[ associate tex_sampler with tex_buffer " << std::endl ;  
    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    tex->setBuffer(deprecated0, deprecated1, texBuffer); 
    std::cout << "] associate tex_sampler with tex_buffer " << std::endl ;  

    std::cout << "[ get texture_id of tex_sampler " << std::endl ;  
    int texture_id = tex->getId(); 
    std::cout << "] get texture_id of tex_sampler " << texture_id << std::endl ;  

    std::cout << "[ creating tex_param_variable " << std::endl ;  
    context["tex_param"]->setInt(optix::make_int4(texture_id, 214, 42, 0 ));
    std::cout << "] creating tex_param_variable " << std::endl ;  
#endif

    std::cout << "[ creating out_buffer " << std::endl ;  
    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 
    outBuffer->setFormat( RT_FORMAT_FLOAT ); 
    outBuffer->setSize(nx, ny, nz); 
    std::cout << "] creating out_buffer " << std::endl ;  

    std::cout << "[ creating out_buffer_variable " << std::endl ;  
    context["out_buffer"]->setBuffer(outBuffer); 
    std::cout << "] creating out_buffer_variable " << std::endl ;  

    std::cout << "[ creating raygen " << std::endl ;  
    optix::Program raygen = context->createProgramFromPTXFile( ptxpath , progname );
    std::cout << "] creating raygen " << std::endl ;  

    std::cout << "[ setting raygen into context " << std::endl ;  
    context->setRayGenerationProgram( entry_point_index, raygen );  
    std::cout << "] setting raygen into context " << std::endl ;  

    std::cout << "[ contextValidate " << std::endl ;  
    context->validate();  
    std::cout << "] contextValidate " << std::endl ;  


    void* tex_data = texBuffer->map() ; 
#ifdef WITH_NPY
    inp->write(tex_data); 
#else
    memcpy( tex_data, inp.data(), sizeof(float)*size ) ;
#endif
    texBuffer->unmap(); 

    std::cout << "[ launch " << std::endl ;  
    context->launch( entry_point_index, nx, ny, nz ); 
    std::cout << "] launch " << std::endl ;  


    void* out_data = outBuffer->map(); 
#ifdef WITH_NPY
    out->read(out_data); 
#else
    memcpy( out.data(), out_data, sizeof(float)*size ) ;
#endif
    outBuffer->unmap(); 


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


