#pragma once

#include <vector>
#include <map>

template <typename T> class NPY ; 

class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventStat
=================

**/


class OKCORE_API OpticksEventStat
{
       typedef std::map<unsigned long long, unsigned>  MQC ;
       typedef std::pair<unsigned long long, unsigned>   PQC ;
       typedef typename std::vector<PQC>  VPQC ;
       static bool PQC_desc( const PQC& a, const PQC& b){ return b.second < a.second ; }

    public:
        static RecordsNPY* CreateRecordsNPY(OpticksEvent* evt);
    public:
        OpticksEventStat(OpticksEvent* evt, unsigned num_cat);
    public:
        void increment(unsigned cat, unsigned long long seqhis_);
        void dump(const char* msg="OpticksEventStat::dump");
    private:
       void init();
       void countTotals();
    private:
       Opticks*                 m_ok ; 
       OpticksEvent*            m_evt ; 
       unsigned                 m_num_cat ; 
       bool                     m_noload ; 

       RecordsNPY*              m_records ; 

       NPY<float>*              m_pho  ; 
       NPY<unsigned long long>* m_seq ;
       unsigned                 m_pho_num ; 
       unsigned                 m_seq_num ; 


       MQC*                     m_counts ; 
       MQC                      m_total ; 
 
       unsigned                 m_totmin ; 


};


#include "OKCORE_TAIL.hh"

 
