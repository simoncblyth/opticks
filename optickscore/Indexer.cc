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

// op --tindexer

#include <numeric>

#include "PLOG.hh"

// npy-
#include "NPY.hpp"

// okc-
#include "Indexer.hh"
#include "Sparse.hh"


template <typename T>
Indexer<T>::Indexer(NPY<T>* seq)
   :
   m_seq(seq),
   m_his(NULL),
   m_mat(NULL),
   m_seqhis(NULL),
   m_seqmat(NULL)
{
}


template <typename T>
void Indexer<T>::indexSequence(const char* seqhis_label, const char* seqmat_label, bool dump, const char* dir)
{
    splitSequence();

    if(dir!=NULL)
    {
        save(dir);
    }

    m_seqhis = new Sparse<T>(seqhis_label, m_his);
    m_seqhis->make_lookup();
    if(dump)
    m_seqhis->dump("indexSequence seqhis");

    m_seqmat = new Sparse<T>(seqmat_label, m_mat);
    m_seqmat->make_lookup();
    if(dump)
    m_seqmat->dump("indexSequence seqmat");
}


template <typename T>
Index* Indexer<T>::getHistoryIndex()
{
    return m_seqhis ? m_seqhis->getIndex() : NULL ; 
}

template <typename T>
Index* Indexer<T>::getMaterialIndex()
{
    return m_seqmat ? m_seqmat->getIndex() : NULL ; 
}


template <typename T>
template <typename S>
void Indexer<T>::applyLookup(S* target)
{
    // fill phosel target with the 

    m_seqhis->template apply_lookup<S>(target,4,0 );
    m_seqmat->template apply_lookup<S>(target,4,1 );
    //  http://stackoverflow.com/questions/3786360/confusing-template-error
}

template <typename T>
void Indexer<T>::splitSequence()
{
    unsigned int num_photons = m_seq->getShape(0);
    typedef std::vector<T> V ;  
    V& seq = m_seq->data();

    LOG(debug) 
              << "Indexer<T>::splitSequence" 
              << " num_photons " << num_photons
              << " m_seq shape " << m_seq->getShapeString()
              << " seq.size " << seq.size()
              ;

    assert( seq.size() == num_photons*2 );
    m_his = NPY<T>::make( num_photons, 1 );
    m_mat = NPY<T>::make( num_photons, 1 );
    m_his->zero();
    m_mat->zero();

    T* seq_v = m_seq->getValues();
    T* his_v = m_his->getValues();
    T* mat_v = m_mat->getValues();

    for(unsigned int i=0 ; i < num_photons ; i++)
    {
       *(his_v + i) = *(seq_v + 2*i + 0) ;
       *(mat_v + i) = *(seq_v + 2*i + 1) ;
    }

    LOG(debug) << "Indexer<T>::splitSequence" 
              << " his " << m_his->getShapeString()
              << " mat " << m_mat->getShapeString()
              ;
}

template <typename T>
void Indexer<T>::save(const char* dir)
{
    m_seq->save(dir,"seq.npy");    
    m_his->save(dir,"his.npy");    
    m_mat->save(dir,"mat.npy");    
}


template class Indexer<unsigned long long> ;
template OKCORE_API void Indexer<unsigned long long>::applyLookup<unsigned char>(unsigned char* target);





