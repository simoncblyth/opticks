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

class GGeoBase ; 
class GScene ; 
class GGeo ;
 
class GGeoLib ;
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GSurLib ; 
class GScintillatorLib ; 

class Composition ; 
class Bookmarks ; 

class OpticksGen ; 
class OpticksGun ; 
class OpticksRun ; 

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

/**

OpticksHub
=============

* Non-viz, hostside intersection of config, geometry and event
* Intended to operate at event level, not below 


Crucial Methods
------------------

GGeoBase* getGGeoBase()
    Effects triangulated/analytic switch by returning downcast 
    of either m_gscene (analytic) or m_ggeo (triangulated)
    based on configured gltf value.


**/

#include "OKGEO_API_EXPORT.hh"

class OKGEO_API OpticksHub {

    // friends using overrideMaterialMapA
       friend class OpMgr ; 
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
       OpticksHub(Opticks* opticks);
   private:
       void init();
       void configure();
       void configureCompositionSize();

       void loadGeometry();
       void configureGeometry(); 
       void configureGeometryTri(); 
       void configureGeometryTriAna(); 

       void configureServer();
       void configureLookupA();
       void overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg);
       void overrideMaterialMapA(const char* jsonA );
   public:
       void add(BCfg* cfg);
   public:
       bool         hasOpt(const char* name);
       bool         isCompute();
   public:
       
   public:
       std::string    getG4GunConfig();
       NPY<float>*    getInputGensteps();
       OpticksEvent*  getG4Event();
       OpticksEvent*  getEvent();
       void createEvent(unsigned tagoffset=0);
       void anaEvent();
   private:
       void configureEvent(OpticksEvent* evt);
   public:
       void setupCompositionTargetting() ;
       void setTarget(unsigned target=0, bool aim=true);
       unsigned getTarget();
   public:
   public:
       // Torchstep is here are it needs geometry for targetting 
       // getter used by CGenerator::makeTorchSource so that cfg4-
       // reuses the same torch 
       TorchStepNPY*        getTorchstep(); 

   public:
       Composition*         getComposition();
       OpticksGeometry*     getGeometry();
   public:
       GGeo*                getGGeo();
       GGeoBase*            getGGeoBase(); // downcast: ( m_gltf ? m_gscene : m_ggeo )
       GGeoBase*            getGGeoBaseAna();
       GGeoBase*            getGGeoBaseTri();
   public:
       GGeoLib*             getGeoLib();
       GMaterialLib*        getMaterialLib();
       GSurfaceLib*         getSurfaceLib();
       GBndLib*             getBndLib();
       GScintillatorLib*    getScintillatorLib();
       GSurLib*             getSurLib();   //  getter triggers creation in GGeo::createSurLib from mesh0

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
       std::string          desc() const ;
   public:
       OpticksRun*          getRun();
       OpticksGen*          getGen();
   public:
       void target();   // point composition at geocenter or the m_evt (last created)
       void configureState(NConfigurable* scene);
       void cleanup();
   private:
       SLog*            m_log ; 
       Opticks*         m_ok ; 
       int              m_gltf ;
       OpticksRun*      m_run ; 
       bool             m_immediate ; 
       OpticksGeometry* m_geometry ; 
       GGeo*            m_ggeo ;  
       GScene*          m_gscene ;  
       Composition*     m_composition ; 

   private:
#ifdef WITH_NPYSERVER
       numpydelegate*              m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       BCfg*                m_cfg ;
       OpticksCfg<Opticks>* m_fcfg ;   
       NState*              m_state ; 
       NLookup*             m_lookup ; 
       Bookmarks*           m_bookmarks ; 

   private:
       OpticksGen*          m_gen ; 
       OpticksGun*          m_gun ; 



};


