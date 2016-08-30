#pragma once

template <typename T> class NPY ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 

class OpticksEvent ; 

#ifdef WITH_OPTIX
class OpEngine ; 
class OpViz ; 
#endif

class OpticksViz ; 


#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API OpticksMgr {
   public:
       OpticksMgr(int argc, char** argv);
   public:
       bool hasOpt(const char* name);
       NPY<float>* loadGenstep();
   public:
       void propagate(NPY<float>* gs);
       void loadPropagation();
       void indexPropagation();
       void visualize();
       void cleanup();
   private:
       void createEvent();
       void init();
       void initGeometry();
       void dbgSeed();
   private:
       Opticks*       m_opticks ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
#ifdef WITH_OPTIX
       OpEngine*      m_ope ; 
       OpViz*         m_opv ; 
#endif
       OpticksEvent*  m_evt ;  // convenience copy of the m_evt in Hub 
       
};

#include "GGV_TAIL.hh"



