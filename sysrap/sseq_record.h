#pragma once
/**
sseq_record.h
===============

Enable selection of photon records with particular histories such as "TO BT AB"
Currently only used from::

    ./sysrap/tests/sseq_record_test.cc


**/

#include "ssys.h"
#include "sstr.h"
#include "spath.h"
#include "sseq_array.h"

struct sseq_record
{
    static constexpr const char* sseq_record__level = "sseq_record__level" ;
    static int level ;
    static constexpr const char* QQ = "TO,CK,SI" ;
    const NP* seq ;
    const NP* record ;
    sseq_array seqa ;

    static bool LooksLikeRecordSeqSelection(const char* q );
    static NP* LoadRecordSeqSelection(const char* _fold, const char* q );

    static sseq_record* Load(const char* fold);
    sseq_record( const NP* _seq, const NP* _record );

    NP* create_record_selection(const char* q_startswith);
};

int sseq_record::level = ssys::getenvint(sseq_record__level,0 );


inline bool sseq_record::LooksLikeRecordSeqSelection(const char* _q )
{
    const char* q = sstr::StartsWith(_q, "$") ? spath::Resolve(_q) : _q ;
    bool q_valid = sstr::StartsWithElem(q, QQ);

    if(!q_valid && level > 0) std::cerr
       << "sseq_record::LooksLikeRecordSeqSelection"
       << " level " << level
       << " _q [" << ( _q ? _q : "-" ) << "]"
       << " q [" << ( q ? q : "-" ) << "]"
       << " QQ " << QQ
       << " q_valid " << ( q_valid ? "YES" : "NO " )
       << "\n"
       ;

    return q_valid ;
}

inline NP* sseq_record::LoadRecordSeqSelection(const char* _fold, const char* _q)
{
    const char* q = sstr::StartsWith(_q, "$") ? spath::Resolve(_q) : _q ;
    bool q_valid = sstr::StartsWithElem(q, QQ);
    assert( q_valid );

    sseq_record* sr = sseq_record::Load(_fold);
    NP* a = sr->create_record_selection(q);
    return a ;
}

inline sseq_record* sseq_record::Load(const char* fold)
{
    const char* seq_path    = spath::Resolve(fold, "seq.npy");
    const char* record_path = spath::Resolve(fold, "record.npy");
    NP* _seq    = NP::LoadIfExists(seq_path);
    NP* _record = NP::LoadIfExists(record_path);

    if(level>0) std::cerr
       << "sseq_record::Load\n"
       << " level " << level
       << " seq_path    " << ( seq_path ? seq_path : "-" ) << "\n"
       << " record_path " << ( record_path ? record_path : "-" ) << "\n"
       << "    _seq " << ( _seq ? _seq->sstr() : "-" ) << "\n"
       << " _record " << ( _record ? _record->sstr() : "-" ) << "\n"
       << "\n"
       ;

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

inline NP* sseq_record::create_record_selection(const char* q )
{
    NP* sel = seqa.create_selection(q);
    NP* record_sel = NP::MakeSelection( record, sel );

    if(level>0) std::cerr
       << "sseq_record::create_record_selection"
       << " level " << level
       << " q " << ( q ? q : "-" )
       << " sel " << ( sel ? sel->sstr() : "-" )
       << " record_sel " << ( record_sel ? record_sel->sstr() : "-" )
       << "\n"
       << "sseq_record::create_record_selection seqa.desc\n"
       << seqa.desc()
       << "\n"
       ;

    return record_sel ;
}


