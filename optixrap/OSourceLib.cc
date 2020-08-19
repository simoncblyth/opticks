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

#include "OConfig.hh"
#include "OSourceLib.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"
#include "PLOG.hh"

OSourceLib::OSourceLib(optix::Context& ctx, GSourceLib* lib)
    : 
    OPropertyLib(ctx, "OSourceLib"),
    m_lib(lib)
{
}

void OSourceLib::convert()
{
    LOG(debug) << "OSourceLib::convert" ;
    NPY<float>* buf = m_lib->getBuffer();
    makeSourceTexture(buf);
}

void OSourceLib::makeSourceTexture(NPY<float>* buf)
{
   // this is fragile, often getting memory errors

    assert(buf && "OSourceLib::makeSourceTexture NULL buffer, try updating geocache first: ggv -G  ? " );

    unsigned int ni = buf->getShape(0);
    unsigned int nj = buf->getShape(1);
    unsigned int nk = buf->getShape(2);
    assert(ni == 1 && nj == GSourceLib::icdf_length && nk == 1);

    unsigned int nx = nj ;
    unsigned int ny = 1 ;

    LOG(debug) << "OSourceLib::makeSourceTexture "
              << " nx " << nx
              << " ny " << ny  
              ;

    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
    
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx, ny );
    upload(optixBuffer, buf);

    optix::TextureSampler tex = m_context->createTextureSampler();
    OConfig::ConfigureSampler(tex, optixBuffer);

    m_context["source_texture"]->setTextureSampler(tex);
    m_context["source_domain"]->setFloat(domain);
}

/*
      (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Validation error: 
      It is forbidden to assign a buffer to both a Buffer and Texture Sampler variable 
      (RTbuffer = 0x0x11ed93b00, RTvariable = source_texture), [4915355])
*/

