#pragma once

#include "string.h"
#include "assert.h"
#include "numpy.hpp"


class GBuffer {
    public:
      GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem)
         :
         m_nbytes(nbytes),     // total number of bytes 
         m_pointer(pointer),   // pointer to the bytes
         m_itemsize(itemsize), // sizeof each item, eg sizeof(gfloat3) = 3*4 = 12
         m_nelem(nelem)        // number of elements for each item, eg 2 or 3 for floats per vertex
      {
      }
      virtual ~GBuffer()
      {
      }
      virtual unsigned int getNumBytes()
      {
          return m_nbytes ;
      }
      virtual void* getPointer()
      {
          return m_pointer ;
      }
      virtual unsigned int getItemSize()
      {
          return m_itemsize ;
      }
      virtual unsigned int getNumElements()
      {
          return m_nelem ;
      }
      virtual unsigned int getNumItems()
      {
          return m_nbytes/m_itemsize ;
      }
      virtual unsigned int getNumElementsTotal()
      {
          return m_nbytes/m_itemsize*m_nelem ;
      }

      /*

          eg 10 float3 vertices, where the item is regarded at the float3 

               NumBytes          10*3*4 = 120 bytes
               ItemSize             3*4 = 12 bytes
               NumElements            3      3 float elements make up the float3
               NumItems              10  =  NumBytes/ItemSize  = 120 bytes/ 12 bytes 
               NumElementsTotal      30  =  NumItems*NumElements = 10*3 

      */ 

      void Summary(const char* msg="GBuffer::Summary");


      template<typename T>
      void save(const char* path);

      template<typename T>
      static GBuffer* load(const char* path);


  protected:
      unsigned int m_nbytes ;
      void*        m_pointer ; 
      unsigned int m_itemsize ;
      unsigned int m_nelem ;

}; 



template<typename T>
inline void GBuffer::save(const char* path)
{
    printf("GBuffer::save path %s \n", path );
    Summary("GBuffer::save");

    void* data = getPointer();
    unsigned int numBytes    = getNumBytes();
    unsigned int numItems    = getNumItems();        
    unsigned int numElements = getNumElements();

    assert(numElements < 5); // elements within an item, eg 3/4 for float3/float4  
    assert(numElements*numItems*sizeof(T) == numBytes ); 

    aoba::SaveArrayAsNumpy<T>( path, numItems, numElements, (T*)data );  
}


template<typename T>
inline GBuffer* GBuffer::load(const char* path)
{
    printf("GBuffer::load path %s \n", path );

    std::vector<T> vdata ;
    int numItems ; 
    int numElements ; 

    aoba::LoadArrayFromNumpy<T>( path, numItems, numElements, vdata );  
     
    assert(numElements < 5);

    unsigned int numBytes = numItems*numElements*sizeof(T);
    unsigned int numValues = numBytes/sizeof(T);
    unsigned int itemSize = numElements*sizeof(T) ; 

    T* tdata = new T[numValues] ;
    memcpy((void*)tdata,  (void*)vdata.data(), numBytes );

    return new GBuffer( numBytes, (void*)tdata,  itemSize, numElements );
}


