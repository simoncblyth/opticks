#pragma once

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

#include "OKCORE_API_EXPORT.hh"
template <typename T>
class OKCORE_API Indexer {
   public:
       Indexer(NPY<T>* seq);
       void indexSequence(const char* seqhis_label, const char* seqmat_label, bool dump=false);

       template <typename S> 
       void applyLookup(S* target);

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


