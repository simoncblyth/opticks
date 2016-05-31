#pragma once

#include <cstdlib>
class NumpyEvt ; 
template <typename T> class NPY ;
template <typename T> class Sparse ;

// CPU only indexer, based on CUDA/Thrust version:
//   
//     opticksop-/OpIndexer
//     thrustrap-/TSparse.hh 
//     thrustrap-/TSparse_.cu
// 
// http://www.boost.org/doc/libs/1_51_0/libs/range/doc/html/range/reference/adaptors/reference/strided.html
//

template <typename T>
class Indexer {
   public:
       Indexer();
       void setEvt(NumpyEvt* evt);
       void indexSequence();
   private:
       void splitSequence();
       void save();
   private:
       NumpyEvt*  m_evt ;  
       NPY<T>*    m_seq ;
       NPY<T>*    m_his ;
       NPY<T>*    m_mat ;
       Sparse<T>* m_seqhis ; 
       Sparse<T>* m_seqmat ; 


};

template <typename T>
inline Indexer<T>::Indexer()
   :
   m_evt(NULL),
   m_seq(NULL),
   m_his(NULL),
   m_mat(NULL),
   m_seqhis(NULL),
   m_seqmat(NULL)
{
}

template <typename T>
inline void Indexer<T>::setEvt(NumpyEvt* evt)
{
    m_evt = evt ;
}



