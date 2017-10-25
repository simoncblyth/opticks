#pragma once

#include <map>
#include <functional>

class NCSG ; 
struct nnode ; 
template <typename T> class NPY ; 

class Opticks ; 
class OpticksEvent ; 


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
   public:
       OpticksEventAna( Opticks* ok, OpticksEvent* evt, NCSG* csg );
       std::string desc();
       void dump(const char* msg="OpticksEventAna::dump");

   private:
       void init();
       void countExcursions();
       void dumpExcursions();

   private:
       Opticks*           m_ok ; 

       float              m_epsilon ;                      
       unsigned long long m_dbgseqhis ;
       unsigned long long m_dbgseqmat ;

       OpticksEvent*            m_evt ; 
       NPY<float>*              m_pho  ; 
       NPY<unsigned long long>* m_seq ;
       unsigned                 m_pho_num ; 
       unsigned                 m_seq_num ; 

       NCSG*         m_csg ;
       nnode*        m_root ;

       typedef std::function<float(float,float,float)> SDF ; 
       typedef std::map<unsigned long long, unsigned>  MQC ;

       SDF                m_sdf ;

       MQC                m_tot ; 
       MQC                m_exc ; 
 

};


#include "OKCORE_TAIL.hh"
 
