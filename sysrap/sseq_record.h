#pragma once
/**
sseq_record.h
===============

Enable selection of photon records with particular histories such as "TO BT AB"
Currently only used from::

    ./sysrap/tests/sseq_record_test.cc


**/

#include "spath.h"
#include "sseq_array.h"

struct sseq_record
{
    const NP* seq ;
    const NP* record ;
    sseq_array seqa ;

    static sseq_record* Load(const char* fold);
    sseq_record( const NP* _seq, const NP* _record );

    NP* create_record_selection(const char* q_startswith);
};


inline sseq_record* sseq_record::Load(const char* fold)
{
    const char* seq_path    = spath::Resolve(fold, "seq.npy");
    const char* record_path = spath::Resolve(fold, "record.npy");
    NP* _seq    = NP::LoadIfExists(seq_path);
    NP* _record = NP::LoadIfExists(record_path);
    return new sseq_record(_seq, _record);
}

inline sseq_record::sseq_record(const NP* _seq, const NP* _record )
    :
    seq(_seq),
    record(_record),
    seqa(seq)
{
}

/**
sseq_record::create_record_selection
-------------------------------------

1. uses sseq_array to create array of sseq photon indices indices
   with histories matching q_startswith

2. applies the seq selection to the record array creating a new
   array with just the selected items

**/

inline NP* sseq_record::create_record_selection(const char* q_startswith)
{
    NP* sel = seqa.create_selection(q_startswith);
    NP* record_sel = NP::MakeSelection( record, sel );
    return record_sel ;
}


