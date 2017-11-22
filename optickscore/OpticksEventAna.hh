#pragma once

#include <vector>
#include <map>

class NCSGList ; 
struct NCSGIntersect ;
//struct nnode ; 
template <typename T> class NPY ; 

class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventStat ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventAna
=================

Splaying GScene::anaEvent into reusability.


Ideas
------

Handle multiple SDFs from NCSGList to check all nodes in 
test geometry ... so cwan infer the nodeindex 
from each photon position (excluding bulk positions, SC, AB )


**/



class OKCORE_API OpticksEventAna
{
   public:
       OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSGList* csglist );

       std::string desc();
       void dump(const char* msg="OpticksEventAna::dump");
       void dumpPointExcursions(const char* msg="OpticksEventAna::dumpPointExcursions");
   private:
       void init();
       void countPointExcursions();
       void checkPointExcursions(); // using the seqmap expectations

   private:
       Opticks*           m_ok ; 

       float              m_epsilon ;                      
       unsigned long long m_dbgseqhis ;
       unsigned long long m_dbgseqmat ;


       unsigned long long m_seqmap_his ;
       unsigned long long m_seqmap_val ;
       bool               m_seqmap_has ; 

       unsigned long long m_seqhis_select ;


       OpticksEvent*            m_evt ; 
      
       NCSGList*                m_csglist ;
       unsigned                 m_tree_num ; 
       NCSGIntersect*           m_csgi ;  


       OpticksEventStat*        m_stat ; 
       RecordsNPY*              m_records ; 

       NPY<float>*              m_pho  ; 
       NPY<unsigned long long>* m_seq ;
       unsigned                 m_pho_num ; 
       unsigned                 m_seq_num ; 




};


#include "OKCORE_TAIL.hh"

