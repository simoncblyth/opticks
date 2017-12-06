#include <cassert>
#include <iostream>
#include "PLOG.hh"

#include "SPairVec.hh"
#include "SMap.hh"
#include "SSeq.hh"
#include "SDigest.hh"


#include "BStr.hh"
#include "BLocSeq.hh"


template <typename T>
const unsigned BLocSeq<T>::MAX_STEP_SEQ = 17  ; 


template <typename T>
BLocSeq<T>::BLocSeq(bool skipdupe)
    :
    m_skipdupe(skipdupe),
    m_global_flat_count(0),
    m_step_flat_count(0),
    m_count_mismatch(0),
    m_seq(skipdupe, true, 1u),
    m_perstep(false),
    m_step_seq(m_perstep ? new BLocSeqDigest<T>[MAX_STEP_SEQ] : NULL),
    m_last_step1(-1)  
{
}

template <typename T>
void BLocSeq<T>::add(const char* loc, int record_id, int step_id )
{
    assert( step_id >= -1 ) ; 

    m_seq.add( loc ) ; 

    unsigned step1 = step_id + 1 ;  // +1 as flat calls happen before G4Ctx::setStep sets the step_id

    // separate loc lists for each step
    if(m_perstep)
    {
        assert( step1 >= 0 && step1 <= MAX_STEP_SEQ && " use --recpoi mode to truncate steps " ); // TODO: why this tops at 16, expected 15  
        m_step_seq[step1].add(loc) ; 
        m_last_step1 = step1 ; 
   } 

    //if(m_global_flat_count < 1000)
    if(step1 > 16)
    {
        LOG(info) 
         << " loc " << std::setw(50) << loc 
         << " record_id " << std::setw(7) << record_id
         << " step_id " <<  std::setw(4) << step_id
         << " global_flat_count " << std::setw(7) << m_global_flat_count 
         << " step_flat_count " << std::setw(7) << m_step_flat_count 
         ; 
    }

    m_step_count[step1]++ ; 
    m_record_count[record_id]++ ; 

    m_global_flat_count++ ; 
    m_step_flat_count++ ; 
}


template <typename T>
void BLocSeq<T>::poststep()
{
/*
    LOG(info) 
         << " global_flat_count " << m_global_flat_count 
         << " step_flat_count " << m_step_flat_count 
         ; 
*/
    m_step_flat_count = 0 ; 
}



// invoked from CRandomEngine::posttrack 
template <typename T>
void BLocSeq<T>::mark(T marker)
{
    m_seq.mark(marker);

    if(m_perstep)
    {
        SSeq<T> mkr(marker); 
        for(int step1=0 ; step1 <= m_last_step1 ; step1++)
        {
            T step_marker = mkr.nibble(step1) ; 
            m_step_seq[step1].mark(step_marker) ; 
        }
    }
}


template <typename T>
void BLocSeq<T>::dump(const char* msg) const 
{
    dumpStepCounts(msg); 
    dumpRecordCounts(msg); 
    m_seq.dumpDigests(msg, false); 
    m_seq.dumpDigests(msg, true); 

    if(m_perstep)
    {
        for(unsigned step1=0 ; step1 < MAX_STEP_SEQ ; step1++)
        {
            LOG(info) << " step1 " << step1 ; 
            m_step_seq[step1].dumpDigests(msg, false);
            m_step_seq[step1].dumpDigests(msg, true);
        }   
    }
}

template <typename T>
void BLocSeq<T>::dumpStepCounts(const char* msg) const 
{ 
    LOG(info) << msg ; 
    typedef std::map<unsigned, unsigned> UMM ; 
    for(UMM::const_iterator it=m_step_count.begin() ; it != m_step_count.end() ; it++ )
    {
        unsigned step_id = it->first ; 
        unsigned count = it->second ; 
        std::cout 
            << std::setw(10) << step_id 
            << std::setw(10) << count
            << std::endl 
            ;
    }
}


template <typename T>
void BLocSeq<T>::dumpRecordCounts(const char* msg) const 
{ 
    LOG(info) << msg ; 
    typedef std::map<unsigned, unsigned> UMM ; 
    for(UMM::const_iterator it=m_record_count.begin() ; it != m_record_count.end() ; it++ )
    {
        unsigned count = it->second ; 
        if(count > 50)
        std::cout 
            << std::setw(10) << it->first 
            << std::setw(10) << count
            << std::endl 
            ;
    }
}


template class BLocSeq<unsigned long long>;

