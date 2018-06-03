#include "OPTICKS_LOG.hh"
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

   NPYBufferSpec spec = buf->getBufferSpec();  
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
   NPYBufferSpec spec = buf->saveToBuffer(vdst) ;   // include the NPY header
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
   NPYBufferSpec spec = buf->saveToBuffer(vdst) ;   // include the NPY header
   assert( spec.bufferByteLength == vdst.size() );

   SSys::xxdump( (char*)vdst.data(), vdst.size(), 16 );  


   NPY<float>* buf2 = NPY<float>::loadFromBuffer( vdst ) ; 
   NPYBufferSpec spec2 = buf2->getBufferSpec() ; 

   buf2->dump(); 

   assert( spec.headerByteLength == spec2.headerByteLength );
   assert( spec.bufferByteLength == spec2.bufferByteLength );

   const char* path = "/tmp/test_loadFromBuffer.npy" ; 
   LOG(info) << " write to " << path ; 

   buf2->save(path); 

}





int main(int argc, char** argv )
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    //test_getBufferSize(); 
    //test_saveToBuffer(); 
    test_loadFromBuffer(); 

    return 0 ; 
}
