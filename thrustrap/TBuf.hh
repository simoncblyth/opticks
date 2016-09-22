#pragma once

#include "CBufSpec.hh"
#include "CBufSlice.hh"

template <typename T> class NPY ; 

#include "THRAP_API_EXPORT.hh" 
class THRAP_API TBuf {
   public:
      TBuf(const char* name, CBufSpec spec );
      void zero();

      void* getDevicePtr() const ;
      unsigned int getNumBytes() const ;
      unsigned int getSize() const ;      // NumItems might be better name
      unsigned int getItemSize() const ;  //  NumBytes/Size

     
      void downloadSelection4x4(const char* name, NPY<float>* npy, bool verbose=false) const ; // selection done on items of size float4x4
      void dump4x4(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const ;

      template <typename T> void downloadSelection(const char* name, NPY<float>* npy, bool verbose=false) const ; // selection done on items of size T
      template <typename T> void fill(T value) const ;
      template <typename T> void upload(NPY<T>* npy) const ;
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


