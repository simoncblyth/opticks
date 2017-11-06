#pragma once

//template <typename T> class NPY ; 
//class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventStat ; 
class OpticksEventDump ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventCompare
======================

**/



class OKCORE_API OpticksEventCompare
{
   public:
       OpticksEventCompare( OpticksEvent* a, OpticksEvent* b);
       void dump(const char* msg="OpticksEventCompare::dump") const ;
       void dumpMatchedSeqHis() const ;
   private:
       Opticks*                 m_ok ; 
       unsigned long long m_dbgseqhis ;
       unsigned long long m_dbgseqmat ;

       OpticksEvent*            m_a ; 
       OpticksEvent*            m_b ; 
      
       OpticksEventStat*        m_as ; 
       OpticksEventStat*        m_bs ; 
      
       OpticksEventDump*        m_ad ; 
       OpticksEventDump*        m_bd ; 




};


#include "OKCORE_TAIL.hh"

