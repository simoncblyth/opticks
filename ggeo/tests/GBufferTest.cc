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

#include "BBufSpec.hh"

#include "Opticks.hh"
#include "GBuffer.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"

/*
   ggv --gbuffer
*/


void test_reshape(Opticks& ok)
{
    const char* idpath = ok.getIdPath() ;
    LOG(info) << "[" 
              << " idpath " << idpath 
              ; 

    GBuffer* buf = GBuffer::load<int>(idpath, "GMergedMesh/1/indices.npy" );
    if(!buf) return ; 
 
    buf->Summary();
    buf->dump<int>("indices", 50);

    buf->reshape(3);
    buf->dump<int>("indices after reshape(3)", 50);

    buf->reshape(1);
    buf->dump<int>("indices after reshape(1)", 50);

    LOG(info) << "]" ; 
}

void test_reshape_slice(Opticks& ok)
{
    const char* idpath = ok.getIdPath() ;
    LOG(info) << "[" ; 
    GBuffer* buf = GBuffer::load<int>(idpath, "GMergedMesh/1/indices.npy" );
    if(!buf) return ; 
    buf->Summary();

    unsigned int nelem = buf->getNumElements();
    buf->reshape(3);
    GBuffer* sbuf = buf->make_slice("0:4") ;
    buf->reshape(nelem);     // return to original 

    sbuf->dump<int>("reshape(3) buffer sliced with 0:4",100);

    sbuf->reshape(1);
    sbuf->dump<int>("after reshape(1)",100);
    LOG(info) << "]" ; 
}


void test_getBufSpec()
{
    unsigned int N = 4 ; 

    float* f = new float[N] ;
    for(unsigned int i=0 ; i < N ; i++ ) f[i] = float(i) ;

    unsigned int nbytes =  sizeof(float)*N ;
    void* pointer = static_cast<void*>(f) ;
    unsigned int itemsize = 4 ; 
    unsigned int nelem = 1 ;

    GBuffer* buf = new GBuffer( nbytes, pointer, itemsize, nelem , "Test");


    BBufSpec* bs = buf->getBufSpec();

    bs->Summary("test_getBufSpec");

    assert(bs->id == -1);
    assert(bs->ptr == pointer);
    assert(bs->num_bytes == nbytes);
    assert(bs->target == 0);
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;  

 
    LOG(info) << argv[0] ; 

    Opticks ok ; 

    LOG(info) << " after ok " ; 

    test_reshape(ok);
    test_reshape_slice(ok);

    test_getBufSpec();

    return 0 ; 
}
