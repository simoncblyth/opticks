#pragma once

template <typename T> class NPY ; 
class Opticks ; 
class OpticksEvent ; 
class OpticksHub ; 
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
   public:
       bool isExit();
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
   private:
       Opticks*       m_opticks ; 
       OpticksHub*    m_hub ; 
       OpticksEvent*  m_evt ; 
#ifdef WITH_OPTIX
       OpEngine*      m_ope ; 
       OpViz*         m_opv ; 
#endif
       OpticksViz*    m_viz ; 
       
};

#include "GGV_TAIL.hh"



