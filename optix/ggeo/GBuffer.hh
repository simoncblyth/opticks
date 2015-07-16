#pragma once

#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "numpy.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



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

      bool isEqual(GBuffer* other)
      {
          return other->getNumBytes() == getNumBytes() && memcmp(other->getPointer(), getPointer(), getNumBytes()) == 0 ;
      }

      float fractionDifferent(GBuffer* other)
      {
          unsigned int numBytes = getNumBytes();
          assert(other->getNumBytes() == numBytes);
          unsigned int numFloats = numBytes/sizeof(float);
          unsigned int n = 0 ;
          float* a = (float*)getPointer();
          float* b = (float*)other->getPointer();

          unsigned int divisor = 39*16 ;        // 624 
          unsigned int qdivisor = divisor/4 ;   // 156

          assert(numFloats % divisor == 0);

          unsigned int numSub = numFloats/divisor ; 
          printf("GBuffer::fractionDifferent numFloats %u divisor %u numFloats/divisor=numSub %u \n", numFloats, divisor, numSub );

          for(unsigned int i=0 ; i < numFloats ; i++)
          {
              unsigned int isub = i/divisor ; 
              unsigned int iquad = (i - isub*divisor)/qdivisor ; 

              if(a[i] != b[i])
              {
                  n+= 1 ; 
                  printf("GBuffer::fractionDifferent i %u isub %u iquad %u  n %u a %10.3f b %10.3f \n", i, isub, iquad, n, a[i], b[i] );  
              }
          }
          return float(n)/float(numFloats) ; 
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

      template<typename T>
      static GBuffer* load(const char* dir, const char* name);



  protected:
      unsigned int m_nbytes ;
      void*        m_pointer ; 
      unsigned int m_itemsize ;
      unsigned int m_nelem ;

}; 



template<typename T>
inline void GBuffer::save(const char* path)
{
    //printf("GBuffer::save path %s \n", path );
    //Summary("GBuffer::save");

    void* data = getPointer();
    unsigned int numBytes    = getNumBytes();
    unsigned int numItems    = getNumItems();        
    unsigned int numElements = getNumElements();

    assert(numElements < 17); // elements within an item, eg 3/4 for float3/float4  
    if(numElements*numItems*sizeof(T) != numBytes )
    {
        LOG(info) << "GBuffer::save " 
                  << " path " << path 
                  << " numBytes " << numBytes 
                  << " numItems " << numItems 
                  << " numElements " << numElements
                  << " numElements*numItems*sizeof(T) " << numElements*numItems*sizeof(T)  
                  ;
    }
    assert(numElements*numItems*sizeof(T) == numBytes ); 

    aoba::SaveArrayAsNumpy<T>( path, numItems, numElements, (T*)data );  
}



template<typename T>
inline GBuffer* GBuffer::load(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256,"%s/%s", dir, name);
    return GBuffer::load<T>(path);
}

template<typename T>
inline GBuffer* GBuffer::load(const char* path)
{
    //printf("GBuffer::load path %s \n", path );

    std::vector<T> vdata ;
    int numItems ; 
    int numElements ; 

    aoba::LoadArrayFromNumpy<T>( path, numItems, numElements, vdata );  // 2d load
    assert(numElements < 17);

    unsigned int numBytes = numItems*numElements*sizeof(T);
    unsigned int numValues = numBytes/sizeof(T);
    unsigned int itemSize = numElements*sizeof(T) ; 

    assert(numValues == vdata.size());

    T* tdata = new T[numValues] ;
    memcpy((void*)tdata,  (void*)vdata.data(), numBytes );

    return new GBuffer( numBytes, (void*)tdata,  itemSize, numElements );
}


