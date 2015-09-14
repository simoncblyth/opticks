#pragma once

#include "OBufSlice.hh"

template <typename T>
class OBufPair {
   public:
      OBufPair( OBufSlice src, OBufSlice dst );
      void seedDestination();
   public:
   private:
      OBufSlice m_src ;   
      OBufSlice m_dst ;   
};


template <typename T>
inline OBufPair<T>::OBufPair(OBufSlice src, OBufSlice dst ) 
   :
   m_src(src),
   m_dst(dst)
{
}


