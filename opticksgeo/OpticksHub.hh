#pragma once

#include <string>
#include <map>

class SLog ; 
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
class NLookup ; 
class NConfigurable ; 
class TorchStepNPY ; 

template <typename> class NPY ;
template <typename> class OpticksCfg ;

#ifdef WITH_NPYSERVER
class numpydelegate ; 
template <typename> class numpyserver ;
#endif

#include "OKGEO_API_EXPORT.hh"


/**

OpticksHub
=============

* Non-viz, hostside intersection of config, geometry and event
* Intended to operate at event level, not below 

**/


class OKGEO_API OpticksHub {

    // friends using overrideMaterialMapA
       friend class OKG4Mgr ; 

    // friends using getEvent
       friend class CG4 ; 
       friend class OpIndexerApp ; 
       friend class OpIndexer ; 
       friend class OpSeeder ; 
       friend class OpZeroer ; 
       friend class OpticksMgr ; 
       friend class OPropagator ; 
       friend class OEngineImp ; 
       friend class OpticksViz ; 
       friend class OpticksIdx ; 
   public:
       OpticksHub(Opticks* opticks, bool config=false);
       void add(BCfg* cfg);
   public:
       void         configure();
       bool         hasOpt(const char* name);
       bool         isCompute();
#ifdef WITH_NPYSERVER
    private:
       void         configureServer();
#endif
   public:
       OpticksEvent* initOKEvent(NPY<float>* gensteps);
       OpticksEvent* loadPersistedEvent();
       void          save();  
   public:
       OpticksEvent* getG4Event();
       OpticksEvent* getOKEvent();
   private:
       OpticksEvent* createG4Event();
       OpticksEvent* createOKEvent();
   private:
       void configureEvent(OpticksEvent* evt);
       OpticksEvent* createEvent(bool ok);
       OpticksEvent* getEvent();   // gets the last created evt, either G4 or OK 
   public:
       void setTarget(unsigned target=0, bool aim=true);
       unsigned getTarget();
   public:
       std::string          getG4GunConfig();
       OpticksEvent*        getZeroEvent(); 
   public:
       // Torchstep is here are it needs geometry for targetting 
       // getter used by CGenerator::makeTorchSource so that cfg4-
       // reuses the same torch 
       TorchStepNPY*        getTorchstep(); 
   private:
       NPY<float>*          loadGenstepFile();
       TorchStepNPY*        makeTorchstep();
   public:
       Composition*         getComposition();
       OpticksGeometry*     getGeometry();
       GGeo*                getGGeo();
       Opticks*             getOpticks();
       OpticksCfg<Opticks>* getCfg();
       std::string          getCfgString();
       NState*              getState();
       NLookup*             getLookup();    // material code translation
       Bookmarks*           getBookmarks();
       NPY<unsigned char>*  getColorBuffer();
       Timer*               getTimer(); 

       OpticksAttrSeq*      getFlagNames();
       OpticksAttrSeq*      getMaterialNames();
       OpticksAttrSeq*      getBoundaryNames();
       std::map<unsigned int, std::string> getBoundaryNamesMap();

   public:
       void loadGeometry();
   public:
       void loadEventBuffers();
       void target();   // point composition at geocenter or the m_evt (last created)
       void configureState(NConfigurable* scene);
       void cleanup();
   private:
       void init();
       void configureCompositionSize();
       void configureLookupA();
       void overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg);
       void setupInputGensteps();
   private:
       SLog*            m_log ; 
       Opticks*         m_ok ; 
       bool             m_immediate ; 
       OpticksGeometry* m_geometry ; 
       GGeo*            m_ggeo ;  
       Composition*     m_composition ; 
   private:
       OpticksEvent*    m_evt ;    // points to last evt created, which is either m_g4evt OR m_okevt 
       OpticksEvent*    m_g4evt ; 
       OpticksEvent*    m_okevt ; 
   private:
       TorchStepNPY*    m_torchstep ; 
       OpticksEvent*    m_zero ;      // objective of zero event is to be available early, and enable "zero" sized event buffers to be used in initialization
#ifdef WITH_NPYSERVER
       numpydelegate*              m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       BCfg*                m_cfg ;
       OpticksCfg<Opticks>* m_fcfg ;   
       NState*              m_state ; 
       NLookup*             m_lookup ; 
       Bookmarks*           m_bookmarks ; 
};


