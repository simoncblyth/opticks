#pragma once

#include <string>
#include <cstring>
#include <vector>

template <typename T> class NPY ;  
template <typename T> class Indexer ;

// translation of thrustrap-/TSparse for non-CPU indexing 

template <typename T>
class Sparse {
   public:
       friend class Indexer<T> ; 
   public:
       Sparse(NPY<T>* npy, const char* label, bool hexkey=true);
   public:
       void count_unique();    
       void dump(const char* msg) const;    
   private:
       void reduce_by_key();
       void sort_by_key();
       std::string dump_(const char* msg) const;    
   private:
       NPY<T>*          m_npy ; 
       const char*      m_label ; 
       bool             m_hexkey ; 
       unsigned int     m_num_unique ;
       std::vector< std::pair<T, int> > m_valuecount ; 

};

template <typename T>
inline Sparse<T>::Sparse(NPY<T>* npy, const char* label, bool hexkey) 
   :
   m_npy(npy),
   m_label(strdup(label)),
   m_hexkey(hexkey),
   m_num_unique(0)
{
}

