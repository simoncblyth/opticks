#pragma once

#include <string>
#include <cstring>
#include <vector>

class Index ; 
template <typename T> class NPY ;  
template <typename T> class Indexer ;

// translation of thrustrap-/TSparse for indexing when CUDA not available

template <typename T>
class Sparse {
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

template <typename T>
inline Sparse<T>::Sparse(const char* label, NPY<T>* source, bool hexkey) 
   :
   m_label(strdup(label)),
   m_source(source),
   m_hexkey(hexkey),
   m_num_unique(0),
   m_num_lookup(0),
   m_index(NULL)
{
   init();
}

template <typename T>
inline Index* Sparse<T>::getIndex()
{
    return m_index ; 
}

