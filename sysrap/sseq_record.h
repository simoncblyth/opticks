#pragma once
/**
sseq_record.h
===============

Enable selection of photon records with particular histories such as "TO BT AB"
Used from::

    sysrap/tests/sseq_record_test.cc
    sysrap/SRecord.h

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
    const char* record_path ;
    const NP* record ;
    sseq_array seqa ;

    static bool LooksLikeRecordSeqSelection(const char* q );
    static NP* LoadRecordSeqSelection(const char* _fold, const char* q );

    static sseq_record* Load(const char* fold);
    sseq_record( const NP* _seq, const NP* _record );
    sseq_record( const NP* _seq, const char* _record_path );

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


/**
sseq_record::LoadRecordSeqSelection
-----------------------------------

Canonical usage from SRecord::LoadArray

**/


inline NP* sseq_record::LoadRecordSeqSelection(const char* _fold, const char* _q)
{
    const char* q = sstr::StartsWith(_q, "$") ? spath::Resolve(_q) : _q ;
    bool q_valid = sstr::StartsWithElem(q, QQ);

    if(!q_valid) std::cerr
        << "sseq_record::LoadRecordSeqSelection"
        << " _fold{" << ( _fold ? _fold : "-" ) << "}"
        << " q{" << ( q ? q : "-" ) << "}"
        << " q_valid " << ( q_valid ? "YES" : "NO ")
        << " -- EXPECTING q TO START WITH ONE OF {" << QQ << "}"
        << "\n"
        ;

    assert( q_valid );

    sseq_record* sr = sseq_record::Load(_fold);
    NP* a = sr->create_record_selection(q);
    return a ;
}

/**
sseq_record::Load
-----------------

This formerly loaded the entire record array - which can be enormous.
Instead just the desired selection of entries from the record array are loaded.

**/


inline sseq_record* sseq_record::Load(const char* fold)
{
    const char* seq_path    = spath::Resolve(fold, "seq.npy");
    const char* record_path = spath::Resolve(fold, "record.npy");
    NP* _seq    = NP::LoadIfExists(seq_path);
    NP* _record = nullptr ;

    if(level>0) std::cerr
       << "sseq_record::Load\n"
       << " level " << level
       << " seq_path    " << ( seq_path ? seq_path : "-" ) << "\n"
       << " record_path " << ( record_path ? record_path : "-" ) << "\n"
       << "    _seq " << ( _seq ? _seq->sstr() : "-" ) << "\n"
       << " _record " << ( _record ? _record->sstr() : "-" ) << "\n"
       << "\n"
       ;

    return new sseq_record(_seq, record_path);
}

inline sseq_record::sseq_record(const NP* _seq, const NP* _record )
    :
    seq(_seq),
    record_path(nullptr),
    record(_record),
    seqa(seq)
{
}

inline sseq_record::sseq_record(const NP* _seq, const char* _record_path )
    :
    seq(_seq),
    record_path(_record_path ? strdup(_record_path) : nullptr),
    record(nullptr),
    seqa(seq)
{
}




/**
sseq_record::create_record_selection
-------------------------------------

This is used from sseq_record::LoadRecordSeqSelection

1. uses sseq_array to create array of sseq photon indices
   with histories matching q_startswith

2. applies the seq selection to the record array creating a new
   array with just the selected items


Old impl loaded the full potentially enormous record array
and then threw away records not matching the history.

New impl loads only the selection of record entries determined from
the much smaller seq array using the new NP::LoadSelection

**/

inline NP* sseq_record::create_record_selection(const char* q )
{
#ifdef OLD_MEMORY_INEFFICIENT_WAY
    NP* sel = seqa.create_selection(q);  // sel contains integer indices selecting items to copy from record
    NP* record_sel = NP::MakeSelection( record, sel );
#else
    NP* sel = seqa.create_selection(q); // side effect is to populate seqa.selection_indices
    NP* record_sel = NP::LoadSelection( record_path, seqa.selection_indices );
#endif

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


