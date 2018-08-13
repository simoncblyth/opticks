#pragma once

#include "CBufSlice.hh"

#include "THRAP_API_EXPORT.hh"

/**
TBufPair
==========

seedDestintion uses strided_range.h for src and dst buffers
and iexpand.h  



**/


template <typename T>
class THRAP_API TBufPair {
   public:
      TBufPair( CBufSlice src, CBufSlice dst, bool verbose=false);
      void seedDestination();
   private:
      CBufSlice m_src ;
      CBufSlice m_dst ;
      bool      m_verbose ; 
};
    

