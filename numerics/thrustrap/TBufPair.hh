#pragma once

#include "CBufSlice.hh"

template <typename T>
class TBufPair {
   public:
      TBufPair( CBufSlice src, CBufSlice dst );
      void seedDestination();
   private:
      CBufSlice m_src ;
      CBufSlice m_dst ;
};
    

template <typename T>
inline TBufPair<T>::TBufPair(CBufSlice src, CBufSlice dst )
   :
   m_src(src),
   m_dst(dst)
{
}

