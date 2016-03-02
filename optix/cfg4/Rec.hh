#pragma once

#include "G4OpBoundaryProcess.hh"
#include <vector>

class CPropLib ; 
class NumpyEvt ; 
class State ; 


class Rec {
   public:
       typedef enum { OK, SKIP_STS } Rec_t ; 
       typedef enum { PRE, POST } Flag_t ; 
   public:
       Rec(CPropLib* clib, NumpyEvt* evt);
   private:
       void init();
   public:
       void add(const State* state); 
       void sequence();
       void Clear();
   public:
       void Dump(const char* msg); 
   public:
       G4OpBoundaryProcessStatus getBoundaryStatus(unsigned int i);
       const State* getState(unsigned int i);
       unsigned int getNumStates();
   public:
       Rec_t getFlagMaterial(unsigned int& flag, unsigned int& material, unsigned int i, Flag_t type );
   public:
       void addFlagMaterial(unsigned int flag, unsigned int material);
       unsigned long long getSeqHis();
       unsigned long long getSeqMat();
   public:
   private:
       CPropLib*                   m_clib ; 
       NumpyEvt*                   m_evt ;  
       unsigned int                m_genflag ;
       std::vector<const State*>   m_states ; 

       unsigned long long          m_seqmat ; 
       unsigned long long          m_seqhis ; 
       unsigned int                m_slot ; 

       unsigned int m_record_max ; 
       unsigned int m_bounce_max ; 
       unsigned int m_steps_per_photon ; 






};

inline Rec::Rec(CPropLib* clib, NumpyEvt* evt)  
   :
    m_clib(clib),
    m_evt(evt), 
    m_genflag(0),
    m_seqhis(0ull),
    m_seqmat(0ull),
    m_slot(0),
    m_record_max(0),
    m_bounce_max(0),
    m_steps_per_photon(0)
{
   init();
}


inline unsigned long long Rec::getSeqHis()
{
    return m_seqhis ; 
}
inline unsigned long long Rec::getSeqMat()
{
    return m_seqmat ; 
}




