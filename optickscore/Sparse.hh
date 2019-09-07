/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
       const char*      m_reldir ; 
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


