#pragma once

template <typename T> class NPY ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class OpticksViz ; 

#ifdef WITH_OPTIX
class OpEngine ; 
class OpViz ; 
#endif

#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API OpticksMgr {
   public:
       OpticksMgr(int argc, char** argv);
       bool hasOpt(const char* name);
   public:
       NPY<float>* loadGenstep();
   public:
       void propagate(NPY<float>* gs);
       void loadPropagation();
       void indexPropagation();
       void visualize();
       void cleanup();
   private:
       void init();
       void initGeometry();
       void dbgSeed();
   private:
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
#ifdef WITH_OPTIX
       OpEngine*      m_ope ; 
       OpViz*         m_opv ; 
#endif
       int            m_placeholder ;  
       
};

#include "GGV_TAIL.hh"

