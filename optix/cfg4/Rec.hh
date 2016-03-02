#pragma once

#include "G4OpBoundaryProcess.hh"
#include <vector>

class CPropLib ; 
class State ; 

class Rec {
   public:
       typedef enum { PRE, POST } Flag_t ; 
   public:
       Rec(CPropLib* clib);
   private:
       void init();
   public:
       void add(const State* state); 
       void Clear();
       void Dump(const char* msg); 
   public:
       const State* getState(unsigned int i);
       unsigned int getNumStates();
       unsigned int getFlag(unsigned int i, Flag_t type);
       unsigned int getPreFlag(unsigned int i);
       unsigned int getPostFlag(unsigned int i);
       G4OpBoundaryProcessStatus getBoundaryStatus(unsigned int i);
   private:
       CPropLib*                   m_clib ; 
       unsigned int                m_genflag ;
       std::vector<const State*>   m_states ; 

};

inline Rec::Rec(CPropLib* clib)  
   :
    m_clib(clib),
    m_genflag(0)
{
   init();
}

inline unsigned int Rec::getPreFlag(unsigned int i)
{
   return getFlag(i, PRE);
}
inline unsigned int Rec::getPostFlag(unsigned int i)
{
   return getFlag(i, POST);
}

