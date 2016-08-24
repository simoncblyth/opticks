#pragma once

class BCfg ; 

class Opticks ; 
class OpticksGeometry ; 
class OpticksEvent ; 
class GGeo ; 
class Composition ; 
class Bookmarks ; 

class NState ; 
class NConfigurable ; 
template <typename> class NPY ;
template <typename> class OpticksCfg ;

#ifdef WITH_NPYSERVER
class numpydelegate ; 
template <typename> class numpyserver ;
#endif

#include "OKGEO_API_EXPORT.hh"

// non-viz, hostside intersection of config, geometry and event
// task: slurp up pieces of App that match the above tagline 

class OKGEO_API OpticksHub {
   public:
       OpticksHub(Opticks* opticks);
       //void setEvent(OpticksEvent* evt);
       void add(BCfg* cfg);
   public:
       void         configure(int argc, char** argv);
       bool         hasOpt(const char* name);
   public:
       Composition*         getComposition();
       GGeo*                getGGeo();
       OpticksEvent*        getEvent();
       OpticksCfg<Opticks>* getCfg();
       std::string          getCfgString();
       NState*              getState();
       Bookmarks*           getBookmarks();
   public:
       void loadGeometry();
       void loadGenstep();
       void loadEvent();
       void targetGenstep();
       void configureViz(NConfigurable* scene);
       void cleanup();
   private:
       void init();
   private:
       NPY<float>* loadGenstepFile();
       NPY<float>* loadGenstepTorch();
   private:
       Opticks*         m_opticks ; 
       OpticksGeometry* m_geometry ; 
       GGeo*            m_ggeo ;  
       Composition*     m_composition ; 
       OpticksEvent*    m_evt ; 

#ifdef WITH_NPYSERVER
       numpydelegate*              m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       BCfg*                m_cfg ;
       OpticksCfg<Opticks>* m_fcfg ;   
       NState*              m_state ; 
       Bookmarks*           m_bookmarks ; 
 
};



