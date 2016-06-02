#pragma once

#include <cstdlib>
template <typename T> class NPY ;
template <typename T> class Sparse ;
class Index ; 

// CPU only indexer, translating from CUDA/Thrust version
// seeks to duplicate results of the GPU indexer
//   
//     opticksop-/OpIndexer
//     thrustrap-/TSparse.hh 
//     thrustrap-/TSparse_.cu
// 

template <typename T>
class Indexer {
   public:
       Indexer(NPY<T>* seq);
       void indexSequence(const char* seqhis_label, const char* seqmat_label);
       template <typename S> void applyLookup(S* target);
       Index* getHistoryIndex();
       Index* getMaterialIndex();
   private:
       void splitSequence();
       void save();
   private:
       NPY<T>*       m_seq ;
   private:
       NPY<T>*    m_his ;
       NPY<T>*    m_mat ;
       Sparse<T>* m_seqhis ; 
       Sparse<T>* m_seqmat ; 


};

template <typename T>
inline Indexer<T>::Indexer(NPY<T>* seq)
   :
   m_seq(seq),
   m_his(NULL),
   m_mat(NULL),
   m_seqhis(NULL),
   m_seqmat(NULL)
{
}

