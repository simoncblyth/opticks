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

// om-;TEST=NPY3Test om-t

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"


const char* TMPDIR = "$TMP/npy/NPY3Test" ; 

void test_getBufferSize()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);

   int num_values = buf->getNumValues();
   assert( num_values == 4 ); 
   std::size_t data_size_expected = num_values*sizeof(float) ;  

   NBufferSpec spec = buf->getBufferSpec();  
   std::size_t data_size = spec.dataSize() ; // bufferSize - headerSize

   const char* path = "test_getBufferSize.npy" ; 
   buf->save(TMPDIR, path); 
   std::size_t file_size = BFile::FileSize(path) ; 

   LOG(info) 
             << " shape " << buf->getShapeString()
             << " num_values " << num_values 
             << " header_size " << spec.headerByteLength
             << " buffer_size " << spec.bufferByteLength
             << " data_size " << data_size
             << " data_size_expected " << data_size_expected
             << " file_size " << file_size
             ;
 
    assert( data_size == data_size_expected ) ; 
    assert( file_size == spec.bufferByteLength ); 
}


void test_saveToBuffer()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);

   std::vector<unsigned char> vdst ;   // buffer gets resized to fit before the save
   NBufferSpec spec = buf->saveToBuffer(vdst) ;   // include the NPY header
   assert( spec.bufferByteLength == vdst.size() );

   SSys::xxdump( (char*)vdst.data(), vdst.size(), 16 );  


   std::string p = BFile::preparePath(TMPDIR, "test_saveToBuffer.npy"); 
   const char* path = p.c_str() ; 
   LOG(info) << " write to " << path ; 
   std::ofstream fp(path, std::ios::out | std::ios::binary ); 
   fp.write( (char*)vdst.data(), vdst.size() ) ;
   fp.close(); 
}


void test_loadFromBuffer()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);

   std::vector<unsigned char> vdst ;   // buffer gets resized to fit before the save
   NBufferSpec spec = buf->saveToBuffer(vdst) ;   // include the NPY header
   assert( spec.bufferByteLength == vdst.size() );

   SSys::xxdump( (char*)vdst.data(), vdst.size(), 16 );  


   NPY<float>* buf2 = NPY<float>::loadFromBuffer( vdst ) ; 
   NBufferSpec spec2 = buf2->getBufferSpec() ; 

   buf2->dump(); 

   assert( spec.headerByteLength == spec2.headerByteLength );
   assert( spec.bufferByteLength == spec2.bufferByteLength );

   const char* path = "test_loadFromBuffer.npy" ; 
   LOG(info) << " write to " << path ; 

   buf2->save(TMPDIR, path); 

}


void test_cast()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);

   NBufferSpec spec = buf->getBufferSpec() ;
   NPYBase* ptr = const_cast<NPYBase*>(spec.ptr) ;  

   LOG(info) << ptr ; 

   NPY<float>*    f = dynamic_cast<NPY<float>*>(ptr) ;
   NPY<double>*   d = dynamic_cast<NPY<double>*>(ptr) ;
   NPY<int>*      i = dynamic_cast<NPY<int>*>(ptr) ;
   NPY<unsigned>* u = dynamic_cast<NPY<unsigned>*>(ptr) ;
 
   assert( f == buf ) ; 
   assert( d == NULL ); 
   assert( i == NULL ); 
   assert( u == NULL ); 

}
void test_add()
{
    LOG(info) << "." ; 
    NPY<float>* planes = NPY<float>::make(0,4);  
   
    glm::vec4 a(1,1,1,1);
    glm::vec4 b(2,1,1,1);
    glm::vec4 c(3,1,1,1);
    
    assert( planes->getNumItems() == 0 );
    planes->add(a); 
    assert( planes->getNumItems() == 1 );
    planes->add(b); 
    assert( planes->getNumItems() == 2 );
    planes->add(c); 
    assert( planes->getNumItems() == 3 );

    planes->dump();
}

void test_u()
{
    LOG(info) << "." ; 
    NPY<unsigned>* idx = NPY<unsigned>::make(1,4);  
    idx->zero();  
    glm::uvec4 id(1,2,3,4) ; 
    idx->setQuad(id, 0) ; 
    idx->save(TMPDIR, "idx.npy"); 
}


void test_setMeta()
{
    LOG(info) << "." ; 
    NPY<unsigned>* idx = NPY<unsigned>::make(1,4);  
    idx->zero();  

    std::string p = BFile::preparePath(TMPDIR, "setMeta.npy") ; 
    const char* path = p.c_str() ; 
    idx->save(path); 
    NPY<unsigned>* idx2 = NPY<unsigned>::load(path) ; 

    const char* key = "loadpath" ; 
    std::string val = path ; 

    idx2->setMeta(key, val);

    std::string val2 = idx2->getMeta<std::string>(key, ""); 
    LOG(info) << "val2 " << val2 ; 

    assert( val2.compare(val) == 0 ) ; 
}





int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv); 

    //test_getBufferSize(); 
    //test_saveToBuffer(); 
    //test_loadFromBuffer(); 

    //test_cast();
    //test_add();
    //test_u();
    test_setMeta();

    return 0 ; 
}

