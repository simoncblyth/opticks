#pragma once

template <typename T> class NPY ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class OpticksViz ; 

#ifdef WITH_OPTIX
class OKPropagator ; 
#endif

#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API OKMgr {
   public:
       OKMgr(int argc, char** argv);
   public:
       void action();
   public:
       void propagate();
       void loadPropagation();
       void visualize();
       void cleanup();
   private:
       void init();
   private:
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
#ifdef WITH_OPTIX
       OKPropagator*  m_propagator ; 
#endif
       int            m_placeholder ;  
       
};

#include "GGV_TAIL.hh"

