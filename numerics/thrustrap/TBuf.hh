#pragma once
#include "stdio.h"
#include "CBufSpec.hh"
#include "CBufSlice.hh"

class TBuf {
   public:
      TBuf(const char* name, CBufSpec spec );

      void* getDevicePtr();

      template <typename T> void dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );
      template <typename T> T  reduce(unsigned int stride, unsigned int begin, unsigned int end=0u );

      CBufSlice slice( unsigned int stride, unsigned int begin, unsigned int end=0u ); 

      void Summary(const char* msg="TBuf::Summary"); 
   private:
      const char* m_name ;
      CBufSpec    m_spec ; 
};


inline TBuf::TBuf(const char* name, CBufSpec spec ) :
        m_name(strdup(name)),
        m_spec(spec)
{
}

inline CBufSlice TBuf::slice( unsigned int stride, unsigned int begin, unsigned int end )
{
    return CBufSlice(m_spec.dev_ptr, m_spec.size, m_spec.num_bytes, stride, begin, end);
}

inline void TBuf::Summary(const char* msg)
{
    printf("%s %s \n", msg, m_name );
}

inline void* TBuf::getDevicePtr()
{
    return m_spec.dev_ptr ; 
}


