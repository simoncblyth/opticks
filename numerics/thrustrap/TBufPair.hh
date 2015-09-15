#pragma once

#include "TBufSlice.hh"

template <typename T>
class TBufPair {
   public:
      TBufPair( TBufSlice src, TBufSlice dst );
      void seedDestination();
   private:
      TBufSlice m_src ;
      TBufSlice m_dst ;
};
    

template <typename T>
inline TBufPair<T>::TBufPair(TBufSlice src, TBufSlice dst )
   :
   m_src(src),
   m_dst(dst)
{
}

