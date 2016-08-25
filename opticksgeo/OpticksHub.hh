#pragma once

#include <string>
#include <map>

class BCfg ; 
class Timer ; 

class Opticks ; 
class OpticksGeometry ; 
class OpticksAttrSeq ; 
class OpticksEvent ; 
class GGeo ; 
class Composition ; 
class Bookmarks ; 

class GItemIndex ; 

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
       OpticksEvent* createEvent();
       void add(BCfg* cfg);
   public:
       void         configure();
       bool         hasOpt(const char* name);
   public:
       Composition*         getComposition();
       GGeo*                getGGeo();
       OpticksEvent*        getEvent();
       Opticks*             getOpticks();
       OpticksCfg<Opticks>* getCfg();
       std::string          getCfgString();
       NState*              getState();
       Bookmarks*           getBookmarks();

       OpticksAttrSeq*      getFlagNames();
       OpticksAttrSeq*      getMaterialNames();
       OpticksAttrSeq*      getBoundaryNames();
       std::map<unsigned int, std::string> getBoundaryNamesMap();

       GItemIndex* makeHistoryItemIndex();
       GItemIndex* makeMaterialItemIndex();
       GItemIndex* makeBoundaryItemIndex();
   public:
       // hostside indexing 
       void indexEvtOld();
       void indexBoundariesHost();
   public:
       void loadGeometry();
       NPY<float>* loadGenstep();
       void loadEvent();
       void targetGenstep();
       void configureViz(NConfigurable* scene);
       void prepareViz();
       void cleanup();
       NPY<unsigned char>* getColorBuffer();
       Timer*              getTimer(); 
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



