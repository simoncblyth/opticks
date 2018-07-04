
#include <cstdio>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "numpy.hpp"

#include "BBufSpec.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "NSlice.hpp"
#include "GBuffer.hh"

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal


bool GBuffer::isEqual(GBuffer* other)
{
    return other->getNumBytes() == getNumBytes() && memcmp(other->getPointer(), getPointer(), getNumBytes()) == 0 ;
}



GBuffer::GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem, const char* name)
         :
         m_nbytes(nbytes),     // total number of bytes 
         m_pointer(pointer),   // pointer to the bytes
         m_itemsize(itemsize), // sizeof each item, eg sizeof(gfloat3) = 3*4 = 12
         m_nelem(nelem),       // number of elements for each item, eg 2 or 3 for floats per vertex or 16 for a 4x4 matrix
         m_name(name ? strdup(name) : NULL),
         m_buffer_id(-1),       // OpenGL buffer Id, set by Renderer on uploading to GPU 
         m_buffer_target(0),
         m_bufspec(NULL)
{
}

const char* GBuffer::getName() const 
{
    return m_name ; 
}

void GBuffer::setName(const char* name)
{
    m_name = name ? strdup( name ) : NULL ;  
}



BBufSpec* GBuffer::getBufSpec()
{   
    if(m_bufspec == NULL)
    {
        int id = getBufferId();
        void* ptr = getPointer();
        unsigned int num_bytes = getNumBytes();
        int target = getBufferTarget() ;  
        m_bufspec = new BBufSpec(id, ptr, num_bytes, target);
    }
    return m_bufspec ; 
}


unsigned int GBuffer::getNumBytes()
{
    return m_nbytes ;
}
void* GBuffer::getPointer()
{
    return m_pointer ;
}
unsigned int GBuffer::getItemSize()
{
    return m_itemsize ;
}
unsigned int GBuffer::getNumElements()
{
    return m_nelem ;
}
unsigned int GBuffer::getNumItems()
{
    return m_nbytes/m_itemsize ;
}
unsigned int GBuffer::getNumElementsTotal()
{
    return m_nbytes/m_itemsize*m_nelem ;
}

void GBuffer::reshape(unsigned int nelem)
{
    if(nelem == m_nelem) return ; 

    bool up = nelem > m_nelem ; 
    if(up) 
    { 
        // reinterpret to a larger "item" with more elements
        assert(nelem % m_nelem == 0);
        unsigned int factor = nelem/m_nelem  ;
        m_nelem = nelem ;
        m_itemsize = m_itemsize*factor ; 
    }
    else
    { 
        // reinterpret to a smaller "item" with less elements  
        assert(m_nelem % nelem == 0);
        unsigned int factor = m_nelem/nelem  ;
        m_nelem = nelem ;
        m_itemsize = m_itemsize/factor ; 
    }
}



// OpenGL related
void GBuffer::setBufferId(int buffer_id)
{
    m_buffer_id = buffer_id  ;
}
int GBuffer::getBufferId()
{
    return m_buffer_id ;
}
void GBuffer::setBufferTarget(int buffer_target)
{
    m_buffer_target = buffer_target  ;
}
int GBuffer::getBufferTarget()
{
    return m_buffer_target ;
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
void GBuffer::save(const char* path_)
{
    //printf("GBuffer::save path %s \n", path );
    //Summary("GBuffer::save");

    std::string path = BFile::FormPath(path_);

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

    if(data == NULL)
    {
        LOG(debug) << "GBuffer::save no data for " << path ; 
    }
    else
    {
        aoba::SaveArrayAsNumpy<T>( path.c_str() , numItems, numElements, (T*)data );  
    }
}



template<typename T>
GBuffer* GBuffer::load(const char* dir, const char* name)
{
    std::string path = BFile::FormPath(dir, name);


/*
    if(!BFile::ExistsFile(path.c_str())
    {
         LOG(warning) << "GBuffer::load"
                      << " FILE DOES NOT EXIST "
                      << " dir " <<  dir
                      << " name " <<  name
                      << " path " << path 
                      ;
          return NULL ;  
 
    }
*/

    return GBuffer::load<T>(path.c_str());
}

template<typename T>
GBuffer* GBuffer::load(const char* path)
{
    //printf("GBuffer::load path %s \n", path );

    std::string name = BFile::Stem(path) ;

    LOG(trace) 
          << " path "  << path 
          << " name "  << name 
           ; 
     // GMeshLib loads 3 buffers for every lvIdx (248 for DYB)           
  

    std::vector<T> vdata ;
    int numItems ; 
    int numElements ; 


    // windows produces obnoxious dialog boxes and do not say where 
    // they occurred when runtime errors are not caught

    try
    {
        aoba::LoadArrayFromNumpy<T>( path, numItems, numElements, vdata );  // 2d load
    }
    catch(const std::runtime_error& /*error*/)
    {
        LOG(warning) << " aoba::LoadArrayFromNumpy FAILED " 
                     << " path " << path 
                     ;        

        return NULL ;   

    }



    LOG(trace) << " path "  << path 
              << " numItems " << numItems 
              << " numElements " << numElements
              ;


    assert(numElements < 17);
    // hmm this is asserting with 3d itransforms (672, 4, 4) in GBufferTest

    unsigned int numBytes = numItems*numElements*sizeof(T);
    unsigned int numValues = numBytes/sizeof(T);
    unsigned int itemSize = numElements*sizeof(T) ; 

    assert(numValues == vdata.size());

    T* tdata = new T[numValues] ;
    memcpy((void*)tdata,  (void*)vdata.data(), numBytes );

    return new GBuffer( numBytes, (void*)tdata,  itemSize, numElements , name.c_str() );
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

    std::string s_name = BStr::concat("s_", m_name, NULL );   

    return new GBuffer( numBytes, (void*)dest, getItemSize(),  getNumElements() , s_name.c_str() )  ; 
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





template GGEO_API void GBuffer::dump<int>(const char* , unsigned int );
template GGEO_API void GBuffer::dump<unsigned int>(const char* , unsigned int);
template GGEO_API void GBuffer::dump<unsigned char>(const char* , unsigned int);
template GGEO_API void GBuffer::dump<float>(const char* , unsigned int);
template GGEO_API void GBuffer::dump<short>(const char* , unsigned int);
template GGEO_API void GBuffer::dump<unsigned long long>(const char* , unsigned int);

template GGEO_API void GBuffer::save<int>(const char* );
template GGEO_API void GBuffer::save<unsigned int>(const char* );
template GGEO_API void GBuffer::save<unsigned char>(const char* );
template GGEO_API void GBuffer::save<float>(const char* );
template GGEO_API void GBuffer::save<short>(const char* );
template GGEO_API void GBuffer::save<unsigned long long>(const char* );

template GGEO_API GBuffer* GBuffer::load<int>(const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned int>(const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned char>(const char* );
template GGEO_API GBuffer* GBuffer::load<float>(const char* );
template GGEO_API GBuffer* GBuffer::load<short>(const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned long long>(const char* );

template GGEO_API GBuffer* GBuffer::load<int>(const char* , const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned int>(const char* , const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned char>(const char* , const char* );
template GGEO_API GBuffer* GBuffer::load<float>(const char* , const char* );
template GGEO_API GBuffer* GBuffer::load<short>(const char* , const char* );
template GGEO_API GBuffer* GBuffer::load<unsigned long long>(const char* , const char* );



