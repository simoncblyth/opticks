#pragma once

#include "CBufSlice.hh"


#include "OXRAP_API_EXPORT.hh"

template <typename T>
class OXRAP_API OBufPair {
   public:
      OBufPair( CBufSlice src, CBufSlice dst );
      void seedDestination();
   public:
   private:
      CBufSlice m_src ;   
      CBufSlice m_dst ;   
};


