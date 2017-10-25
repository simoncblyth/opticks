#pragma once

#include <vector>
#include <map>
#include <functional>

class NCSGList ; 
struct nnode ; 
template <typename T> class NPY ; 

class RecordsNPY ; 

class Opticks ; 
class OpticksEvent ; 
class OpticksEventDump ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventAna
=================

Splaying GScene::anaEvent into reusability.


Ideas
------

Handle multiple SDFs from NCSGList to check all nodes in 
test geometry ... so can infer the nodeindex 
from each photon position (excluding bulk positions, SC, AB )


**/



class OKCORE_API OpticksEventAna
{
       typedef std::function<float(float,float,float)> SDF ; 
       typedef std::map<unsigned long long, unsigned>  MQC ;
       typedef std::pair<unsigned long long, unsigned>   PQC ;
       typedef typename std::vector<PQC>  VPQC ;

       static bool PQC_desc( const PQC& a, const PQC& b){ return b.second < a.second ; }

   public:
       OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSGList* csglist );
       std::string desc();
       void dump(const char* msg="OpticksEventAna::dump");

   private:
       void init();
       void countTotals();
       void countExcursions();
       void dumpExcursions();
       void dumpStepByStepCSGExcursions();
       void dumpStepByStepCSGExcursions(unsigned photon_id );

   private:
       Opticks*           m_ok ; 

       float              m_epsilon ;                      
       unsigned long long m_dbgseqhis ;
       unsigned long long m_dbgseqmat ;

       OpticksEvent*            m_evt ; 
      

       NCSGList*                m_csglist ;
       unsigned                 m_tree_num ; 

       SDF*                     m_sdflist ; 
       MQC*                     m_surf ; 
       MQC                      m_total ; 
 
       OpticksEventDump*        m_dmp ; 
       RecordsNPY*              m_records ; 

       NPY<float>*              m_pho  ; 
       NPY<unsigned long long>* m_seq ;
       unsigned                 m_pho_num ; 
       unsigned                 m_seq_num ; 




};


#include "OKCORE_TAIL.hh"

