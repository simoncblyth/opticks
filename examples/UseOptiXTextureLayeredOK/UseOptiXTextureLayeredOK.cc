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
#include "OPTICKS_LOG.hh"
#include "NPY.hpp"


void Make2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPY<float>* inp)
{
    unsigned nd = inp->getDimensions(); 
    assert( nd == 3 );    

    LOG(info) << " inp " << inp->getShapeString() ;  

    const unsigned ni = inp->getShape(0);  // number of texture layers
    const unsigned nj = inp->getShape(1); 
    const unsigned nk = inp->getShape(2); 

    const unsigned layers = ni ; 
    const unsigned height = nj ; 
    const unsigned width = nk ; 

    unsigned bufferdesc = RT_BUFFER_INPUT | RT_BUFFER_LAYERED ; 
    //  If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 
    texBuffer->setFormat( RT_FORMAT_FLOAT ); 
    texBuffer->setSize(width, height, layers);      // 3rd depth arg is number of layers

    // attempt at using mapEx failed, so upload all layers at once 

    bool exfill = false ; 
/**
exfill:true always giving exception::

    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Already mapped 
    (Details: Function "RTresult bufferMap(RTbuffer, unsigned int, unsigned int, void *, void **)" caught exception: Buffer is already mapped)
**/
    if(exfill)
    {
        for(unsigned i=0 ; i < ni ; i++)
        {
            LOG(info) << "[ map i " << i ; 
            void* tex_data = texBuffer->map(i) ; 
            inp->write_item_(tex_data, i); 
            texBuffer->unmap(i); 
            LOG(info) << "] map i " << i ; 
        }
    }
    else
    {
        void* tex_data = texBuffer->map() ; 
        inp->write(tex_data); 
        texBuffer->unmap(); 
    }


    std::cout << "[ creating tex_sampler " << std::endl ;  
    optix::TextureSampler tex = context->createTextureSampler(); 

    //RTwrapmode wrapmode = RT_WRAP_REPEAT ; 
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ; 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ; 
    tex->setWrapMode(0, wrapmode);
    tex->setWrapMode(1, wrapmode);
    //tex->setWrapMode(2, wrapmode);   corresponds to layer?

    RTfiltermode filtermode = RT_FILTER_NEAREST ;  // RT_FILTER_LINEAR 
    RTfiltermode minification = filtermode ; 
    RTfiltermode magnification = filtermode ; 
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

    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    tex->setBuffer(deprecated0, deprecated1, texBuffer); 

    unsigned tex_id = tex->getId() ; 
    context[param_key]->setInt(optix::make_int4(ni, nj, nk, tex_id));
}

//  $HOME/opticks_refs/Earth_Albedo_8192_4096.jpg


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    bool index_test = true ;  

    const int nx = 4u ; 
    const int ny = 4u ; 
    const int nz = 4u ;
 
    NPY<float>* inp = NPY<float>::make(nx, ny, nz ); 
    if(index_test) inp->fillIndexFlat(); // fill with flattened index values, for debug 
    inp->dump();

    inp->setMeta<float>("xmin", 0.f); 
    inp->setMeta<float>("xmax", 360.f); 
    inp->setMeta<float>("ymin", 0.f); 
    inp->setMeta<float>("ymax", 180.f); 

    inp->save("$TMP/UseOptiXTextureLayeredOK/inp.npy"); 


    const char* cmake_target = "UseOptiXTextureLayeredOK" ;
    const char* cu_name = "UseOptiXTextureLayeredOK.cu" ; 
    const char* ptxpath = OKConf::PTXPath( cmake_target, cu_name ); 
    const char* progname = "readWrite" ;

    std::cout 
        << " ptxpath: [" << ptxpath << "]" 
        << " progname: [" << progname << "]" 
        << std::endl
        ; 

    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1); 
    context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
    context->setPrintEnabled(1);  
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);
    unsigned entry_point_index = 0u ; 

    Make2DLayeredTexture(context, "tex_param", "tex_domain", inp);   


    NPY<float>* out = NPY<float>::make(nx, ny, nz); 
    out->zero();
    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 
    outBuffer->setFormat( RT_FORMAT_FLOAT ); 
    outBuffer->setSize(nx, ny, nz); 
    context["out_buffer"]->setBuffer(outBuffer); 


    optix::Program raygen = context->createProgramFromPTXFile( ptxpath , progname );
    context->setRayGenerationProgram( entry_point_index, raygen );  
    context->validate();  

    std::cout << "[ launch " << std::endl ;  
    context->launch( entry_point_index, nx, ny, nz ); 
    std::cout << "] launch " << std::endl ;  

    void* out_data = outBuffer->map(); 
    out->read(out_data); 
    outBuffer->unmap(); 

    out->dump();
    out->save("$TMP/UseOptiXTextureLayeredOK/out.npy"); 

    if(index_test)
    {
        unsigned mismatch = out->compareWithIndexFlat() ; 
        assert( mismatch == 0 );  
    }

    return 0 ; 
}


