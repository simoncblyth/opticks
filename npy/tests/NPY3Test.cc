// om-;TEST=NPY3Test om-t

#include "OPTICKS_LOG.hh"
#include "SBufferSpec.hh"
#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"

void test_getBufferSize()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);

   int num_values = buf->getNumValues();
   assert( num_values == 4 ); 
   std::size_t data_size_expected = num_values*sizeof(float) ;  

   NBufferSpec spec = buf->getBufferSpec();  
   std::size_t data_size = spec.dataSize() ; // bufferSize - headerSize

   const char* path = "$TMP/test_getBufferSize.npy" ; 
   buf->save(path); 
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

   const char* path = "/tmp/test_saveToBuffer.npy" ; 
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

   const char* path = "/tmp/test_loadFromBuffer.npy" ; 
   LOG(info) << " write to " << path ; 

   buf2->save(path); 

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


int main(int argc, char** argv )
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    //test_getBufferSize(); 
    //test_saveToBuffer(); 
    //test_loadFromBuffer(); 

    //test_cast();
    test_add();

    return 0 ; 
}

