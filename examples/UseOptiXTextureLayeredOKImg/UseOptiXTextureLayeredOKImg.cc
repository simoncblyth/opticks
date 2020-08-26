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
#include "OFormat.hh"
#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "ImageNPY.hpp"

/**
TODO:

1. tighten up, make it higher level by pulling off common stuff into OConfig for example
2. for generality need to return domain range info as well as the id 

   * hmm perhaps use NPY metadata for this (it survives persisting)


3. expt with realistic 2d theta-phi layered spherical texture : especially wrt the wrapping mode, passing domain range 
4. apply 2d theta-phi layered texture to some instanced geometry and shade based on it : eg make a variety of PPM beach-balls  

**/


#include "UseOptiXTextureLayeredOKImg.h"


#ifdef TEX_BUFFER_CHECK
template <typename T>
void UploadBuffer(optix::Context& context, const char* buffer_key, const NPYBase* inp)
{
    unsigned nd = inp->getDimensions(); 
    assert( nd == 4 );    

    const unsigned ni = inp->getShape(0);  // number of texture layers
    const unsigned nj = inp->getShape(1);  // width 
    const unsigned nk = inp->getShape(2);  // height
    const unsigned nl = inp->getShape(3);  // components
    assert( nl < 5 );   

    unsigned bufferdesc = RT_BUFFER_INPUT  ; 
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 

    RTformat format = OFormat::TextureFormat<T>(nl);
    texBuffer->setFormat( format ); 
    texBuffer->setSize(nj, nk, ni);    

    void* tex_data = texBuffer->map() ; 
    inp->write_(tex_data); 
    texBuffer->unmap(); 

    context[buffer_key]->setBuffer(texBuffer); 
}
#endif


template <typename T>
void Upload2DLayeredTexture(optix::Context& context, const char* param_key, const char* domain_key, const NPYBase* inp)
{
    unsigned nd = inp->getDimensions(); 
    assert( nd == 4 );    

    const unsigned ni = inp->getShape(0);  // number of texture layers
    const unsigned nj = inp->getShape(1);  // width 
    const unsigned nk = inp->getShape(2);  // height
    const unsigned nl = inp->getShape(3);  // components
    assert( nl < 5 );   

    float xmin = inp->getMeta<float>("xmin", "0.") ; 
    float xmax = inp->getMeta<float>("xmax", "1.") ; 
    float ymin = inp->getMeta<float>("ymin", "0.") ; 
    float ymax = inp->getMeta<float>("ymax", "1.") ; 
    context[domain_key]->setFloat(optix::make_float4(xmin, xmax, ymin, ymax));

    LOG(info) 
        << " xmin " << xmin 
        << " xmax " << xmax 
        << " ymin " << ymin 
        << " ymax " << ymax 
        ;

    unsigned bufferdesc = RT_BUFFER_INPUT | RT_BUFFER_LAYERED ; 
    //  If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
    optix::Buffer texBuffer = context->createBuffer(bufferdesc); 

    RTformat format = OFormat::Get<T>(nl);
    texBuffer->setFormat( format ); 
    texBuffer->setSize(nj, nk, ni);      // 3rd depth arg is number of layers

    // attempt at using mapEx failed, so upload all layers at once 
    void* tex_data = texBuffer->map() ; 
    inp->write_(tex_data); 
    texBuffer->unmap(); 

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

    unsigned deprecated0 = 0 ; 
    unsigned deprecated1 = 0 ; 
    tex->setBuffer(deprecated0, deprecated1, texBuffer); 

    unsigned tex_id = tex->getId() ; 
    context[param_key]->setInt(optix::make_int4(ni, nj, nk, tex_id));
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cmake_target = "UseOptiXTextureLayeredOKImg" ;
    const char* cu_name = "UseOptiXTextureLayeredOKImg.cu" ; 
    const char* ptxpath = OKConf::PTXPath( cmake_target, cu_name ); 
    const char* progname = "readWrite" ;
    const char* dir = "$TMP/UseOptiXTextureLayeredOKImg" ; 


    const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ;
    bool yflip = false ; 
    unsigned ncomp_ = 4 ; 

    const char* config = "" ; 
    NPY<unsigned char>* inp = ImageNPY::LoadPPM(path, yflip, ncomp_, config) ; 
    assert( inp->getDimensions() == 3 ); 
    LOG(info) << " original inp (height, width, ncomp)  " << inp->getShapeString() ; 

    unsigned layers = 1 ; 
    unsigned height = inp->getShape(0);  
    unsigned width = inp->getShape(1);  
    unsigned ncomp = inp->getShape(2);  

    inp->reshape(layers,height,width,ncomp) ; // unsure re height<->width layout 
    LOG(info) << " after reshape inp (layers,height,width,ncomp) " << inp->getShapeString()  ; 
 
    inp->setMeta<float>("xmin", 0.f); 
    inp->setMeta<float>("xmax", 360.f); 
    inp->setMeta<float>("ymin", 0.f); 
    inp->setMeta<float>("ymax", 180.f); 
    inp->save(dir,"inp.npy"); 

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

#ifdef TEX_BUFFER_CHECK
    UploadBuffer<unsigned char>(context, "tex_buffer", inp);   
#else
    Upload2DLayeredTexture<unsigned char>(context, "tex_param", "tex_domain", inp);   
#endif

    NPY<unsigned char>* out = NPY<unsigned char>::make(layers, height, width, ncomp); 
    out->zero();
    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 

    RTformat format = OFormat::Get<unsigned char>(ncomp);
    outBuffer->setFormat(format); 
    outBuffer->setSize(height, width, layers); 
    context["out_buffer"]->setBuffer(outBuffer); 

    optix::Program raygen = context->createProgramFromPTXFile( ptxpath , progname );
    context->setRayGenerationProgram( entry_point_index, raygen );  
    context->validate();  

    std::cout << "[ launch " << std::endl ;  
    context->launch( entry_point_index, height, width, layers ); 
    std::cout << "] launch " << std::endl ;  

    void* out_data = outBuffer->map(); 
    out->read(out_data); 
    outBuffer->unmap(); 

    out->dump();
    out->save(dir,"out.npy"); 


    return 0 ; 
}


