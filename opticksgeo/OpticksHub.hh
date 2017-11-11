#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>


class SLog ; 
class BCfg ; 
class Timer ; 

class Opticks ; 
class OpticksGeometry ; 
class OpticksAttrSeq ; 
class OpticksEvent ; 

class GenstepNPY  ; 

class GGeoBase ; 
class GScene ; 
class GGeo ;
class GGeoTest ;

class GPmt ; 
class GMergedMesh ;
class GItemIndex ; 
 
class GGeoLib ;
class GMaterialLib ; 
class GSurfaceLib ; 
class GBndLib ; 
class GSourceLib ; 
class GScintillatorLib ; 
class GNodeLib ;
class GPmtLib ;

class Composition ; 
class Bookmarks ; 

class OpticksGen ; 
class OpticksGun ; 
class OpticksRun ; 
class OpticksAim ; 


class NCSG ; 
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

   public:
       int getErr() const ;
       unsigned getSourceCode() const ; 
  private:
       void init();
       void setErr(int err);
       void configure();
       void configureCompositionSize();

       void loadGeometry();
       GGeoTest* createTestGeometry(GGeoBase* basis);

       void registerGeometry(); 
       void configureGeometry(); 
       void configureGeometryTri(); 
       void configureGeometryTriAna(); 
       void configureGeometryTest(); 
   private:
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
       void         dumpSolids(unsigned cursor, GMergedMesh* mm, const char* msg="OpticksHub::dumpSolids" );  
   public:
       std::string    getG4GunConfig();
       NPY<float>*    getInputGensteps() const ;
       NPY<float>*    getInputPhotons();
       OpticksEvent*  getG4Event();
       OpticksEvent*  getEvent();
       void createEvent(unsigned tagoffset=0);
       void anaEvent();
   private:
       void configureEvent(OpticksEvent* evt);
  public:
       // via OpticksAim
       void            setupCompositionTargetting();
       void            target();   // point composition at geocenter or the m_evt (last created)
       void            setTarget(unsigned target=0, bool aim=true);
       unsigned        getTarget();
  public:
       // Torchstep is here are it needs geometry for targetting 
       // getter used by CGenerator::makeTorchSource so that cfg4-
       // reuses the same torch 
       TorchStepNPY*        getTorchstep(); 
       GenstepNPY*          getGenstepNPY();

   public:
       Composition*         getComposition();
       OpticksGeometry*     getGeometry();

   public:
       GGeoBase*            getGGeoBase() const ; // downcast of the encumbent: GGeoTest/GScene/GGeo 

   public:
       // via encumbent 
       glm::mat4            getTransform(int index);

   private:
       GGeoBase*            getGGeoBaseAna() const ;
       GGeoBase*            getGGeoBaseTri() const ;
       GGeoBase*            getGGeoBasePrimary() const ;  // either Ana or Tri 
       GGeoBase*            getGGeoBaseTest() const ;    // downcast of GGeoTest
   private:
       //GGeo*                getGGeo();
   private:
       friend class CTestDetector ; 
       GGeoTest*            getGGeoTest();  
   public:
       NCSG*                findEmitter() const ; 

   public:
       // hmm hub could be a GGeoBase itself
       // all the below libs etc are dispensed from one of 3 possible GGeoBase, 
       // namely GGeo/GScene/GGeoTest 
       const char*          getIdentifier(); 
       GMergedMesh*         getMergedMesh( unsigned index );

       GGeoLib*             getGeoLib();       //  meshes :   until very recently direct from GGeo
       GMaterialLib*        getMaterialLib();  //  materials
       GSurfaceLib*         getSurfaceLib();   //  surfaces
       GBndLib*             getBndLib();       //  boundaries

       GPmtLib*             getPmtLib();       //   partlist? analytic PMT   
       GScintillatorLib*    getScintillatorLib();
       GSourceLib*          getSourceLib();
       GNodeLib*            getNodeLib() ; 

   public:
       Opticks*             getOpticks();
       OpticksCfg<Opticks>* getCfg();
       std::string          getCfgString();
       NState*              getState();
       NLookup*             getLookup();    // material code translation
       Bookmarks*           getBookmarks();
       NPY<unsigned char>*  getColorBuffer();
       Timer*               getTimer(); 

       OpticksAttrSeq*      getFlagNames();

       std::string          desc() const ;
   public:
       OpticksRun*          getRun();
       OpticksGen*          getGen();
   public:
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
       OpticksAim*          m_aim ;
 
       GGeoTest*            m_geotest ; 
       int                  m_err ; 



};


