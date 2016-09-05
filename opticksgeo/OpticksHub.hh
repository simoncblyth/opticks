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
class NLookup ; 
class NConfigurable ; 
class TorchStepNPY ; 
class G4StepNPY ; 

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
       void setGensteps(NPY<float>* gs);   // does checks and defines G4StepNPY 
       NPY<float>*          getGensteps();  
       NPY<float>*          getNopsteps();  // updated when new G4 event is created
   public:
       TorchStepNPY*        getTorchstep(); // needs geometry for targetting 
       G4StepNPY*           getG4Step();    // created in translateGenstep
   private:
       void                 translateGensteps(NPY<float>* gs);  // into Opticks lingo
       NPY<float>*          loadGenstepFile();
       TorchStepNPY*        makeTorchstep();
   public:
       Composition*         getComposition();
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
       NPY<float>*      m_nopsteps ;
       NPY<float>*      m_gensteps ;
       TorchStepNPY*    m_torchstep ; 
       G4StepNPY*       m_g4step ;
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



