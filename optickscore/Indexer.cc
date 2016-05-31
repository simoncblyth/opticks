// op --tindexer

#include "Indexer.hh"
#include "Sparse.hh"

// npy-
#include "NumpyEvt.hpp"
#include "NLog.hpp"

#include <numeric>


template <typename T>
void Indexer<T>::indexSequence()
{
    splitSequence();
    save();

    m_seqhis = new Sparse<T>(m_his, "his");
    m_seqmat = new Sparse<T>(m_mat, "mat");

    m_seqhis->count_unique();
    m_seqmat->count_unique();

    m_seqhis->dump("indexSequence seqhis");
    m_seqmat->dump("indexSequence seqmat");

}

template <typename T>
void Indexer<T>::splitSequence()
{
    m_seq = m_evt->getSequenceData();

    unsigned int num_photons = m_seq->getShape(0);
    typedef std::vector<T> V ;  
    V& seq = m_seq->data();

    LOG(info) 
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

    LOG(info) << "Indexer<T>::splitSequence" 
              << " his " << m_his->getShapeString()
              << " mat " << m_mat->getShapeString()
              ;
}

template <typename T>
void Indexer<T>::save()
{
    m_seq->save("/tmp/seq.npy");    
    m_his->save("/tmp/his.npy");    
    m_mat->save("/tmp/mat.npy");    
}



template class Indexer<unsigned long long> ;
