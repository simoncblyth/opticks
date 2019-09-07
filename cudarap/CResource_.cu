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

#include <cstdlib>
#include <cassert>

#include <cuda_gl_interop.h>
#include "CResource.hh"
#include "helper_cuda.h"   // for checkCudaErrors

#include "cfloat4x4.h"


struct CResourceImp 
{
    unsigned      buffer_id ; 
    size_t        bufsize  ; 
    unsigned      flags ; 
    cudaStream_t  stream ; 
    struct cudaGraphicsResource*  resource ;
    void*         dev_ptr ;   

    CResourceImp(unsigned buffer_id_, unsigned flags_, cudaStream_t stream_) 
        : 
        buffer_id(buffer_id_),
        bufsize(0),
        flags(flags_),  
        stream(stream_),
        resource(NULL),
        dev_ptr(NULL)
    {
    }
    
    // HMM : stream_ arg was previously ignored with steam(NULL) : Changed to taking copy of stream arg Jun 26, 2016 

    const char* getFlagDescription()
    {
        const char* ret(NULL);
        switch(flags)
        {
            case cudaGraphicsMapFlagsNone:         ret="cudaGraphicsMapFlagsNone: Default; Assume resource can be read/written " ; break ;
            case cudaGraphicsMapFlagsReadOnly:     ret="cudaGraphicsMapFlagsReadOnly: CUDA will not write to this resource " ; break ; 
            case cudaGraphicsMapFlagsWriteDiscard: ret="cudaGraphicsMapFlagsWriteDiscard: CUDA will only write to and will not read from this resource " ; break ;  
        }
        return ret ;
    }

    void registerBuffer()
    {
        //printf("Resource::registerBuffer %d : %s \n", buffer_id, getFlagDescription() );
        checkCudaErrors( cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags) );
    }

    void unregisterBuffer()
    {
        //printf("Resource::unregisterBuffer %d \n", buffer_id );
        checkCudaErrors( cudaGraphicsUnregisterResource(resource) );
    }


    void* mapGLToCUDA() 
    {
        checkCudaErrors( cudaGraphicsMapResources(1, &resource, stream) );
        checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &bufsize, resource) );
        //printf("Resource::mapGLToCUDA bufsize %lu dev_ptr %p \n", bufsize, dev_ptr );
        return dev_ptr ; 
    }

    void unmapGLToCUDA()
    {
        //printf("Resource::unmapGLToCUDA\n");
        checkCudaErrors( cudaGraphicsUnmapResources(1, &resource, stream));
    } 

    void streamSync()
    {
        //printf("Resource::streamSync\n");
        checkCudaErrors( cudaStreamSynchronize(stream));
    }

};


void CResource::init()
{
    unsigned flgs(0) ;
    switch(m_access)
    {
        case RW: flgs = cudaGraphicsMapFlagsNone         ;break;
        case  R: flgs = cudaGraphicsMapFlagsReadOnly     ;break;
        case  W: flgs = cudaGraphicsMapFlagsWriteDiscard ;break;
    }
    //cudaStream_t stream1 ; 
    //cudaStreamCreate ( &stream1) ;
    m_imp = new CResourceImp(m_buffer_id, flgs, (cudaStream_t)0  );
}

void CResource::streamSync()
{
    m_imp->streamSync();
}

template <typename T>
CBufSpec CResource::mapGLToCUDA()
{
    m_mapped = true ; 
    m_imp->registerBuffer();
    m_imp->mapGLToCUDA();
    unsigned int size = m_imp->bufsize/sizeof(T) ;
    //printf("CResource::mapGLToCUDA buffer_id %d imp.bufsize %lu sizeof(T) %lu size %d \n", m_buffer_id, m_imp->bufsize, sizeof(T), size );
    return CBufSpec( m_imp->dev_ptr, size, m_imp->bufsize );  // number of items only defined when decide on item size
}
void CResource::unmapGLToCUDA()
{
    m_mapped = false ; 
    //printf("CResource::unmapGLToCUDA\n");
    m_imp->unmapGLToCUDA();
    m_imp->unregisterBuffer();
}



template CUDARAP_API CBufSpec CResource::mapGLToCUDA<unsigned char>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<unsigned int>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<unsigned long long>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<short>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<int>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<float>();
template CUDARAP_API CBufSpec CResource::mapGLToCUDA<cfloat4x4>();



