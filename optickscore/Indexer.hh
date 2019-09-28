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
       void indexSequence(const char* seqhis_label, const char* seqmat_label, bool dump=false, const char* dir=NULL);

       template <typename S> 
       void applyLookup(S* target);

       Index* getHistoryIndex();
       Index* getMaterialIndex();
   private:
       void splitSequence();
       void save(const char* dir);
   private:
       NPY<T>*       m_seq ;
   private:
       NPY<T>*    m_his ;
       NPY<T>*    m_mat ;
       Sparse<T>* m_seqhis ; 
       Sparse<T>* m_seqmat ; 


};


