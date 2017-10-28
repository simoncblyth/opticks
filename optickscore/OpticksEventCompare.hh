#pragma once

//template <typename T> class NPY ; 
//class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventStat ; 

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
       void dump(const char* msg="OpticksEventCompare::dump");
   private:
       Opticks*                 m_ok ; 

       OpticksEvent*            m_a ; 
       OpticksEvent*            m_b ; 
      
       OpticksEventStat*        m_as ; 
       OpticksEventStat*        m_bs ; 


};


#include "OKCORE_TAIL.hh"

