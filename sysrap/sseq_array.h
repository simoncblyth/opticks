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
    std::vector<int64_t> selection_indices ;

    sseq_array( const NP* seq );
    sseq_array( const char* lines );

    NP*  create_selection(const char* q_spec);
    void find_selection_indices(const char* q_spec);
    void find_selection_indices_ending(  const char* q_endswith);
    void find_selection_indices_starting(const char* q_startswith);

    std::string desc() const ;
};

inline sseq_array::sseq_array(const NP* seq)
{
    NPX::VecFromArray<sseq>(qq, seq );
}

inline sseq_array::sseq_array(const char* lines)
{
    std::stringstream ss;
    ss.str(lines)  ;
    std::string str ;
    while (std::getline(ss, str, '\n'))
    {
        if(str.empty()) continue ;
        const char* l = str.c_str();

        sseq sq ;
        sq.zero();
        sq.set_history(l);
        qq.push_back(sq);
    }
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

Slice suffix can be used to restrict entries eg::

   "TO BT BR BT SA[::10]"
   "TO BT BR BT SA[:1000]"

**/


inline NP* sseq_array::create_selection(const char* q_spec)
{
    find_selection_indices(q_spec);
    NP* sel = NPX::Make<int64_t>(selection_indices);
    return sel ;
}

inline void sseq_array::find_selection_indices(const char* q_spec)
{
    bool has_wildcard_0 = q_spec && strlen(q_spec) > 1 && q_spec[0] == '*' ;
    if(has_wildcard_0)
    {
        find_selection_indices_ending(q_spec);
    }
    else
    {
        find_selection_indices_starting(q_spec);
    }
}


inline void sseq_array::find_selection_indices_ending(const char* q_spec)
{
    assert(q_spec[0] == '*');
    const char* q_spec_1 = q_spec + 1;

    selection_indices.clear();

    char* q_endswith = nullptr ;
    char* sli = nullptr ;
    bool has_slice = sstr::prefix_suffix(&q_endswith, &sli, "[", q_spec_1);

    NP_slice<int64_t>* slice = nullptr ;
    if(has_slice)
    {
        slice = new NP_slice<int64_t>();
        int rc = slice->parse(sli, true);
        if(rc!=0) std::cerr
            << "sseq_array::find_selection_indices_ending"
            << " FAILED TO PARSE sli {" << ( sli ? sli : "-" ) << "}\n"
            ;
        assert(rc == 0);
    }

    std::vector<std::string> q_ews ;
    sstr::Split(q_endswith, ',', q_ews );
    assert( q_ews.size() == 1 ); // only one endswith string supported currently
    const char* q_ew = q_ews[0].c_str();

    unsigned flag = OpticksPhoton::AbbrevToFlag(q_ew);
    assert( flag != 0 );
    unsigned ffs_flag = FFS(flag);

    size_t nqq = qq.size(); // number of seq histories

    int64_t sliced_count = 0 ;
    int64_t unsliced_count = 0 ;

    for(size_t i=0 ; i < nqq ; i++)
    {
        const sseq& q = qq[i] ;
        bool match = q.last_nibble() == ffs_flag ;
        if(match)
        {
            bool select = slice ? slice->contains(unsliced_count) : true ;
            unsliced_count += 1 ;

            if(select)
            {
                sliced_count += 1 ;
                selection_indices.push_back(i);
            }
        }
    }
}


inline void sseq_array::find_selection_indices_starting(const char* q_spec)
{
    selection_indices.clear();

    // split eg "TO BT BR BT SA,TO BT BR BR BT SA[:1000000]"  into "TO BT BR BT SA,TO BT BR BR BT SA" and "[:1000000]"
    char* q_startswith = nullptr ;
    char* sli = nullptr ;
    bool has_slice = sstr::prefix_suffix(&q_startswith, &sli, "[", q_spec );

    NP_slice<int64_t>* slice = nullptr ;
    if(has_slice)
    {
        slice = new NP_slice<int64_t>();
        int rc = slice->parse(sli, true);
        if(rc!=0) std::cerr
            << "sseq_array::find_selection_indices"
            << " FAILED TO PARSE sli {" << ( sli ? sli : "-" ) << "}\n"
            ;
        assert(rc == 0);
    }

    std::vector<std::string> q_sws ;
    sstr::Split(q_startswith, ',', q_sws );

    size_t nqq = qq.size(); // number of seq histories

    int64_t sliced_count = 0 ;
    int64_t unsliced_count = 0 ;

    for(size_t i=0 ; i < nqq ; i++)
    {
        const sseq& q = qq[i] ;
        std::string his = q.seqhis_();

        int match = 0 ; // supports an OR of comma delimited selections
        for(int j=0 ; j < int(q_sws.size()) ; j++)
        {
            const char* q_sw = q_sws[j].c_str();
            bool startswith = 0==strncmp(his.c_str(), q_sw, strlen(q_sw));
            if(startswith) match += 1;
        }


        if(match > 0)
        {
            bool select = slice ? slice->contains(unsliced_count) : true ;
            unsliced_count += 1 ;

            if(select)
            {
                sliced_count += 1 ;
                selection_indices.push_back(i);
            }
        }
    }
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
