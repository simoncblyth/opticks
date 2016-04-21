#include "GBuffer.hh"

#include <cstdio>
#include <cstring>
#include <cassert>

#include <iostream>
#include <iomanip>

#include "numpy.hpp"
#include "NSlice.hpp"

#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


bool GBuffer::isEqual(GBuffer* other)
{
    return other->getNumBytes() == getNumBytes() && memcmp(other->getPointer(), getPointer(), getNumBytes()) == 0 ;
}


void GBuffer::Summary(const char* msg)
{
    LOG(info) << std::left << std::setw(30) << msg << std::right
              << " BufferId " << std::setw(4) << getBufferId()
              << " BufferTarget " << std::setw(4) << getBufferTarget()
              << " NumBytes " << std::setw(7) << getNumBytes()
              << " ItemSize " << std::setw(7) << getItemSize()
              << " NumElements " << std::setw(7) << getNumElements()
              << " NumItems " << std::setw(7) << getNumItems()
              << " NumElementsTotal " << std::setw(7) << getNumElementsTotal()
              ;
}


float GBuffer::fractionDifferent(GBuffer* other)
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






template<typename T>
void GBuffer::save(const char* path)
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
GBuffer* GBuffer::load(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256,"%s/%s", dir, name);
    return GBuffer::load<T>(path);
}

template<typename T>
GBuffer* GBuffer::load(const char* path)
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


GBuffer* GBuffer::make_slice(const char* slice_)
{
    NSlice* slice = slice_ ? new NSlice(slice_) : NULL ;
    return make_slice(slice);
}

GBuffer* GBuffer::make_slice(NSlice* slice)
{
    unsigned int ni = getNumItems(); 
    if(!slice) 
    {
        slice = new NSlice(0, ni, 1);
        LOG(warning) << "GBuffer::make_slice NULL slice, defaulting to full copy " << slice->description() ;
    }
    //unsigned int nk = getNumElements(); 
    unsigned int size = getItemSize(); 

    char* src = (char*)getPointer();
    unsigned int count = slice->count();

    LOG(info) << "GBuffer::make_slice from " 
              << ni << " -> " << count 
              << " slice " << slice->description() ;

    assert(count <= ni);

    unsigned int numBytes = count*size ; 
    char* dest = new char[numBytes] ;    
    unsigned int offset = 0 ; 
    for(unsigned int i=slice->low ; i < slice->high ; i+=slice->step)
    {
        memcpy( (void*)(dest + offset),(void*)(src + size*i), size ) ;
        offset += size ; 
    }
    return new GBuffer( numBytes, (void*)dest, getItemSize(),  getNumElements())  ; 
} 



template<typename T>
void GBuffer::dump(const char* msg, unsigned int limit)
{
    Summary(msg);

    unsigned int ni = getNumItems(); 
    unsigned int sz = getItemSize(); 
    unsigned int nk = getNumElements(); 
    char* ptr = (char*)getPointer();

    for(unsigned int i=0 ; i < std::min(ni, limit) ; i++)
    {
        T* v = (T*)(ptr + sz*i) ; 
        for(unsigned int k=0 ; k < nk ; k++)
        {   
            if(k%nk == 0) std::cout << std::endl ; 

            if(k==0) std::cout << "(" <<std::setw(3) << i << ") " ;
            std::cout << " " << std::fixed << std::setprecision(3) << std::setw(10) << *(v+k) << " " ;  
        }   
   }
   std::cout << std::endl ; 
}





template void GBuffer::dump<int>(const char* , unsigned int );
template void GBuffer::dump<unsigned int>(const char* , unsigned int);
template void GBuffer::dump<unsigned char>(const char* , unsigned int);
template void GBuffer::dump<float>(const char* , unsigned int);
template void GBuffer::dump<short>(const char* , unsigned int);
template void GBuffer::dump<unsigned long long>(const char* , unsigned int);

template void GBuffer::save<int>(const char* );
template void GBuffer::save<unsigned int>(const char* );
template void GBuffer::save<unsigned char>(const char* );
template void GBuffer::save<float>(const char* );
template void GBuffer::save<short>(const char* );
template void GBuffer::save<unsigned long long>(const char* );

template GBuffer* GBuffer::load<int>(const char* );
template GBuffer* GBuffer::load<unsigned int>(const char* );
template GBuffer* GBuffer::load<unsigned char>(const char* );
template GBuffer* GBuffer::load<float>(const char* );
template GBuffer* GBuffer::load<short>(const char* );
template GBuffer* GBuffer::load<unsigned long long>(const char* );

template GBuffer* GBuffer::load<int>(const char* , const char* );
template GBuffer* GBuffer::load<unsigned int>(const char* , const char* );
template GBuffer* GBuffer::load<unsigned char>(const char* , const char* );
template GBuffer* GBuffer::load<float>(const char* , const char* );
template GBuffer* GBuffer::load<short>(const char* , const char* );
template GBuffer* GBuffer::load<unsigned long long>(const char* , const char* );




