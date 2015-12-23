#pragma once
#include "stdio.h"
#include "CBufSpec.hh"
#include "CBufSlice.hh"

template <typename T> class NPY ; 

class TBuf {
   public:
      TBuf(const char* name, CBufSpec spec );
      void zero();

      void* getDevicePtr() const ;
      unsigned int getNumBytes() const ;
      unsigned int getSize() const ;

      template <typename T> void download(NPY<T>* npy) const ;
      template <typename T> void repeat_to(TBuf* other, unsigned int stride, unsigned int begin, unsigned int end, unsigned int repeats) const ;
      template <typename T> void dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const ;
      template <typename T> void dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const ;
      template <typename T> T  reduce(unsigned int stride, unsigned int begin, unsigned int end=0u ) const ;

      CBufSlice slice( unsigned int stride, unsigned int begin=0u, unsigned int end=0u ) const ; 

      void Summary(const char* msg="TBuf::Summary") const ; 
   private:
      const char* m_name ;
      CBufSpec    m_spec ; 
};


inline TBuf::TBuf(const char* name, CBufSpec spec ) :
        m_name(strdup(name)),
        m_spec(spec)
{
}

inline CBufSlice TBuf::slice( unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    if(end == 0u) end = m_spec.size ;  
    return CBufSlice(m_spec.dev_ptr, m_spec.size, m_spec.num_bytes, stride, begin, end);
}

inline void TBuf::Summary(const char* msg) const 
{
    printf("%s %s \n", msg, m_name );
}

inline void* TBuf::getDevicePtr() const 
{
    return m_spec.dev_ptr ; 
}
inline unsigned int TBuf::getNumBytes() const 
{
    return m_spec.num_bytes ; 
}
inline unsigned int TBuf::getSize() const 
{
    return m_spec.size ; 
}



