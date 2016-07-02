#pragma once

#include "CBufSlice.hh"

#include "THRAP_API_EXPORT.hh"
template <typename T>
class THRAP_API TBufPair {
   public:
      TBufPair( CBufSlice src, CBufSlice dst );
      void seedDestination();
   private:
      CBufSlice m_src ;
      CBufSlice m_dst ;
};
    

