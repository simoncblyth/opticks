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

#include "NPY.hpp"
#include "GScintillatorLib.hh"
#include "OConfig.hh"
#include "OScintillatorLib.hh"

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal


const plog::Severity OScintillatorLib::LEVEL = debug ; 


OScintillatorLib::OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib)
    : 
    OPropertyLib(ctx, "OScintillatorLib"),
    m_lib(lib),
    m_placeholder(NPY<float>::make(1,4096,1))
{
    m_placeholder->zero(); 
}

/*

From OKTest running off a DYB geocache with scintillators::

    2018-08-14 15:01:21.142 ERROR [29988] [OScintillatorLib::makeReemissionTexture@78] 
        OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 1 nj 4096 nk 1 step 0.000244141 empty 0

*/


void OScintillatorLib::convert(const char* slice)
{
    NPY<float>* buf = m_lib->getBuffer();
    unsigned ni = buf ? buf->getShape(0) : 0 ; 

    LOG(verbose) << "OScintillatorLib::convert" 
               << " from " << ( buf ? buf->getShapeString() : "-" ) 
               << " ni " << ni 
               ;

    if( ni == 0) 
    {
        LOG(LEVEL) << " empty GScintillatorLib buffer : creating placeholder reemission texture " ; 
        makeReemissionTexture(m_placeholder);
    }
    else if( ni == 1 )
    { 
        makeReemissionTexture(buf);
    }
    else if( ni > 1 && slice )
    { 
        NPY<float>* slice_buf = buf->make_slice(slice) ;

        LOG(verbose) << "OScintillatorLib::convert" 
                  << " sliced buffer with " << slice
                  << " from " << buf->getShapeString()
                  << " to " << slice_buf->getShapeString()
                  ;
 
        makeReemissionTexture(slice_buf);
    }

    LOG(verbose) << "OScintillatorLib::convert DONE" ;
}




void OScintillatorLib::makeReemissionTexture(NPY<float>* buf)
{
    if(!buf)
    {
       LOG(fatal) << "OScintillatorLib::makeReemissionTexture MISSING BUFFER " ;
       LOG(fatal) << " you probably need to populate the geocache for the current geometry selection " ;

    } 
    assert(buf) ;  

    unsigned ni = buf->getShape(0);
    unsigned nj = buf->getShape(1);
    unsigned nk = buf->getShape(2);

    bool empty = ni == 0 ;
     
    unsigned nx = 4096 ; 
    unsigned ny = 1 ; 

    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );

    LOG(LEVEL) << "OScintillatorLib::makeReemissionTexture "
              << " nx " << nx
              << " ny " << ny  
              << " ni " << ni  
              << " nj " << nj  
              << " nk " << nk
              << " step " << step
              << " empty " << empty
              ;

    if(empty)
    {
        LOG(error) << "OScintillatorLib::makeReemissionTexture no scintillators, skipping " ;
        return ;   
    }
 
    optix::Buffer buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx, ny );

    upload(buffer, buf);

    optix::TextureSampler tex = m_context->createTextureSampler();
    OConfig::ConfigureSampler(tex, buffer);

    m_context["reemission_texture"]->setTextureSampler(tex);
    m_context["reemission_domain"]->setFloat(domain);

    LOG(verbose) << "OScintillatorLib::makeReemissionTexture DONE " ; 
}




