#pragma once
#include "OBufBase.hh"
// NB implementation in OBuf_.cu as requires nvcc compilation


// anything not using or related to template type should go in OBufBase

// TODO: avoid the code duplication between TBuf and OBuf ?
//  hmm OBuf could contain a TBuf ?
//  
//
//
// Using a templated class rather than templated member functions 
// has the advantage of only having to explicitly instanciate the class::
//
//    template class OBuf<optix::float4> ;
//    template class OBuf<optix::uint4> ;
//    template class OBuf<unsigned int> ;
//
// as opposed to having to explicly instanciate all the member functions.
//
// But when want differently typed "views" of the 
// same data it seems more logical to used templated member functions.
//

#include "OXRAP_API_EXPORT.hh"


class OXRAP_API OBuf : public OBufBase {
   public:
      OBuf( const char* name, optix::Buffer& buffer); 

   public:
      template <typename T>
      void dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );

      template <typename T>
      void dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );

      template <typename T>
      T reduce(unsigned int stride, unsigned int begin, unsigned int end=0u );

};


