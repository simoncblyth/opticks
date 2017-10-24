#pragma once

#include <map>
#include <functional>

class NCSG ; 
struct nnode ; 
template <typename T> class NPY ; 

class OpticksEvent ; 


#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksEventAna
=================

Splaying GScene::anaEvent into reusability.


**/



class OKCORE_API OpticksEventAna
{
   public:
       OpticksEventAna( OpticksEvent* evt, NCSG* csg );
       std::string desc();
       void dump(const char* msg="OpticksEventAna::dump");

   private:
       void init();
       void countExcursions();
       void dumpExcursions();

   private:
       OpticksEvent*            m_evt ; 
       NPY<float>*              m_pho  ; 
       NPY<unsigned long long>* m_seq ;
       unsigned                 m_pho_num ; 
       unsigned                 m_seq_num ; 


       NCSG*         m_csg ;
       nnode*        m_root ;
       std::function<float(float,float,float)> m_sdf ;
       float         m_epsilon ;                      
 
       typedef std::map<unsigned long long, unsigned> MQC ;
       MQC          m_tot ; 
       MQC          m_exc ; 
 

};


#include "OKCORE_TAIL.hh"
 
