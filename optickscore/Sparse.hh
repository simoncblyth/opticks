#pragma once

#include <string>
#include <vector>

class Index ; 
template <typename T> class NPY ;  
template <typename T> class Indexer ;

// translation of thrustrap-/TSparse for indexing when CUDA not available

#include "OKCORE_API_EXPORT.hh"

template <typename T>
class OKCORE_API Sparse {
   public:
       friend class Indexer<T> ; 
       typedef std::pair<T, int> P ;
       enum { SPARSE_LOOKUP_N=32 };
   public:
       Sparse(const char* label, NPY<T>* source, bool hexkey=true);
       void make_lookup();
       template <typename S> void apply_lookup(S* target, unsigned int stride, unsigned int offset );
       Index* getIndex();
       void dump(const char* msg) const;    
   private:
       void init();
       unsigned int count_value(const T value) const ;    
       void count_unique();    
       void update_lookup();
       void reduce_by_key(std::vector<T>& data);
       void sort_by_key();
       void populate_index(Index* index);
       std::string dump_(const char* msg, bool slowcheck=false) const;    
   private:
       const char*      m_label ; 
       NPY<T>*          m_source ; 
       bool             m_hexkey ; 
       unsigned int     m_num_unique ;
       std::vector< std::pair<T, int> > m_valuecount ; 
   private:
       unsigned int      m_num_lookup ; // truncated m_num_unique to be less than SPARSE_LOOKUP_N 
       std::vector<T>    m_lookup ; 
       Index*            m_index ; 
       T                 m_sparse_lookup[SPARSE_LOOKUP_N];

};


