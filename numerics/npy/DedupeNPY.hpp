#pragma once
#include "NPY.hpp"

// hmm doing this properly post-cache is difficult 
// as involves not just vertices and faces, but loadsa other buffers too 


template <typename T>
class DedupeNPY {
   public:
      DedupeNPY(NPY<T>* src);

      NPY<T>* getSrc();
      NPY<T>* getDst();

   private:
      NPY<T>* m_src ; 
      NPY<T>* m_dst ; 
};


template <typename T>
inline DedupeNPY<T>::DedupeNPY(NPY<T>* src) 
    : 
      m_src(src), 
      m_dst(NULL) 
{
}

template <typename T>
inline NPY<T>* DedupeNPY<T>::getSrc()
{
    return m_src ; 
}

template <typename T>
inline NPY<T>* DedupeNPY<T>::getDst()
{
    return m_dst ; 
}






