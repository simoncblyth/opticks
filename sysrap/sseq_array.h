#pragma once
/**
sseq_array.h
=============

Facilitate history selection using seq array

This is used from::

    sseq_record.h

**/

#include "NPX.h"
#include "sseq.h"
#include "sstr.h"

struct sseq_array
{
    std::vector<sseq> qq ;
    sseq_array( const NP* seq );
    NP* create_selection(const char* q_startswith);
    std::string desc() const ;
};

inline sseq_array::sseq_array(const NP* seq)
{
    NPX::VecFromArray<sseq>(qq, seq );
}

/**
sseq_array::create_selection
-----------------------------

Create array of int64_t indices into the source seq array
with histories that match the argument, eg::

   "TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT SD"
   "TO BT BT BT SA,TO BT BT BT EC"

A comma can be used to delimit multiple histories that are
individually used with the OR over all histories selection
being returned.

**/


inline NP* sseq_array::create_selection(const char* q_startswith)
{
    std::vector<std::string> q_sws ;
    sstr::Split(q_startswith, ',', q_sws );

    std::vector<int64_t> vv ;
    int nqq = int(qq.size());
    for(int i=0 ; i < nqq ; i++)
    {
        const sseq& q = qq[i] ;
        std::string his = q.seqhis_();

        int match = 0 ;
        for(int j=0 ; j < int(q_sws.size()) ; j++)
        {
            const char* q_sw = q_sws[j].c_str();
            bool startswith = 0==strncmp(his.c_str(), q_sw, strlen(q_sw));
            if(startswith) match += 1;
        }
        if(match > 0) vv.push_back(i);
    }
    NP* sel = NPX::Make<int64_t>(vv);
    return sel ;
}

/**
sseq_array::desc
------------------

Return summary of histories in the seq array.

**/


inline std::string sseq_array::desc() const
{
    int nqq = int(qq.size());
    int edge = 10 ;
    std::stringstream ss ;
    ss << "[sseq_array::desc " << nqq << "\n"  ;
    for(int i=0 ; i < nqq ; i++)
    {
        if( ( i < edge)  || ((nqq - i) < edge) ) ss << std::setw(8) << i << "[" << qq[i].seqhis_() << "]\n" ;
    }
    ss << "]sseq_array::desc\n"  ;
    std::string str = ss.str();
    return str ;
}
