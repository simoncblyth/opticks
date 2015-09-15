#pragma once

#include "CBufSlice.hh"

template <typename T>
class OBufPair {
   public:
      OBufPair( CBufSlice src, CBufSlice dst );
      void seedDestination();
   public:
   private:
      CBufSlice m_src ;   
      CBufSlice m_dst ;   
};


template <typename T>
inline OBufPair<T>::OBufPair(CBufSlice src, CBufSlice dst ) 
   :
   m_src(src),
   m_dst(dst)
{
}


