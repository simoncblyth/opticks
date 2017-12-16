#pragma once


#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

#include "BLocSeqDigest.hh"


template <typename T> 
class BRAP_API BLocSeq
{
        static const unsigned MAX_STEP_SEQ ; 
    public:
        BLocSeq(bool skipdupe);

        void add(const char* loc, int record_id, int step_id); 
        void postStep();
        void mark(T marker); 

    public:
        void dump(const char* msg) const ; 
    private:
        void dumpRecordCounts(const char* msg) const ; 
        void dumpStepCounts(const char* msg) const ; 

    private:
        bool                          m_skipdupe ; 
        unsigned                      m_global_flat_count ; 
        unsigned                      m_step_flat_count ; 
        unsigned                      m_count_mismatch; 

        std::map<unsigned, unsigned>  m_record_count ; 
        std::map<unsigned, unsigned>  m_step_count ; 

        BLocSeqDigest<T>              m_seq ; 

        bool                          m_perstep ; 
        BLocSeqDigest<T>*             m_step_seq ; 
        int                           m_last_step1 ; 


};

#include "BRAP_TAIL.hh"

