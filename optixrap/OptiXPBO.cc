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


// http://rickarkin.blogspot.tw/2012/03/use-pbo-to-share-buffer-between-cuda.html

class CudaOptiXPBO
{
  protected:
      unsigned int                    _pbo;
      size_t                          _devBufSize;
      void*                           _devBufAddr;
      struct cudaGraphicsResource    *_pbo_resource;

  public:
       CudaOptiXPBO(){};
      ~CudaOptiXPBO() {};
 
      void createCudaPbo(size_t sizeByte)
      {
          glGenBuffers( 1, &_pbo );
          glBindBuffer( GL_PIXEL_UNPACK_BUFFER, _pbo );
          glBufferData( GL_PIXEL_UNPACK_BUFFER, sizeByte, NULL, GL_STREAM_DRAW );
          glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

          cutilSafeCall(cudaGraphicsGLRegisterBuffer(&_pbo_resource, _pbo, cudaGraphicsMapFlagsNone));
      }

      void releaseCudaPbo()
      {
          cutilSafeCall(cudaGraphicsUnregisterResource(_pbo_resource));

          glDeleteBuffers(1, &_pbo);
          _pbo = 0;
      }

      unsigned int getPbo() { return _pbo; }

      void* preCudaOp()
      {
          cutilSafeCall(cudaGraphicsMapResources(1, &_pbo_resource, 0));    
          cutilSafeCall(cudaGraphicsResourceGetMappedPointer(_devBufAddr, _devBufSize, _pbo_resource));
          return _devBufAddr;
      }

      void postCudaOp()
      {
           cutilSafeCall(cudaGraphicsUnmapResources(1, &_pbo_resource, 0));
      }

};  // CudaOptiXPBO



/*

struct MyData
{
    float var;
};


unsigned int elemNum = 1024;  // number of elements.
CudaOptixPBO _coPBO;
_coPBO.createCudaPbo(ElementNumber * sizeofElement);
_coPBO.getPbo());

optix::Buffer buffer = optixContext->createBufferFromGLBO(RT_BUFFER_INPUT, buffer->setFormat(RT_FORMAT_USER);
buffer->setElementSize(sizeof(MyData));
buffer->setSize(elemNum);

optixContext["OptixBufferName"]->setBuffer(buffer);


*/





