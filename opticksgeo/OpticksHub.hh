#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

class SCtrl ; 
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
class FlightPath ; 

class OpticksGen ; 
class OpticksRun ; 
class OpticksAim ; 


class NCSG ; 
class NState ; 
class NLookup ; 
class NConfigurable ; 
class TorchStepNPY ; 

template <typename> class NPY ;
template <typename> class OpticksCfg ;

#include "SCtrl.hh"

#ifdef OPTICKS_NPYSERVER
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

class OKGEO_API OpticksHub : public SCtrl {

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
       static const plog::Severity LEVEL ; 
   public:
       OpticksHub(Opticks* ok); 

   public:
       int getErr() const ;
       void setCtrl(SCtrl* ctrl);

   public:
       // SCtrl
       void command(const char* cmd);  // no longer in chain, moved to OpticksViz
  private:
       void init();
       void setErr(int err);
       void configure();
       void configureCompositionSize();

       void loadGeometry();
       void adoptGeometry();  // when operating from a passed in GGeo
       GGeoTest* createTestGeometry(GGeoBase* basis);

       void registerGeometry(); 
   private:
       // TODO: eliminate Tri/Ana mess : as now always have both 
       void configureGeometry(); 
       void configureGeometryTri(); 
       void configureGeometryTriAna(); 
       void configureGeometryTest(); 
   private:
       //void configureGeometryPrep(); moved to Opticks
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
       void         dumpVolumes(unsigned cursor, GMergedMesh* mm, const char* msg="OpticksHub::dumpVolumes" );  
   public:
      // from m_gen
       unsigned       getSourceCode() const ; 
       NPY<float>*    getInputPhotons() const ;
       NPY<float>*    getInputGensteps() const ;
       NPY<float>*    getInputPrimaries() const ;
       TorchStepNPY*  getTorchstep() const ;   // Torchstep is here as needs geometry for targetting 
       GenstepNPY*    getGenstepNPY() const ;
       std::string    getG4GunConfig() const ;
  public:

   public:
       OpticksEvent*  getG4Event();
       OpticksEvent*  getEvent();
       void createEvent(unsigned tagoffset=0);
       void anaEvent();
   private:
       void configureEvent(OpticksEvent* evt);
       void anaEvent(OpticksEvent* evt);
  public:
       // via OpticksAim
       void            setupCompositionTargetting();
       void            target();   // point composition at geocenter or the m_evt (last created)
       void            setTarget(unsigned target=0, bool aim=true);
       unsigned        getTarget();
   public:
       Composition*         getComposition();
       OpticksGeometry*     getGeometry();

   public:
       GGeo*                getGGeo() const ;
       GGeoBase*            getGGeoBase() const ; // downcast of the encumbent: GGeoTest/GScene/GGeo 

   public:
       // via encumbent 
       glm::mat4            getTransform(int index);

   private:
       GGeoBase*            getGGeoBaseAna() const ;  // downcast GScene, didnt that go the way of the Dodo ?  TODO:eliminate  
       GGeoBase*            getGGeoBaseTri() const ;
       GGeoBase*            getGGeoBasePrimary() const ;  // either Ana or Tri 
       GGeoBase*            getGGeoBaseTest() const ;    // downcast of GGeoTest
   private:
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

       GGeoLib*             getGeoLib();       //  meshes 
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
       Bookmarks*           getBookmarks() const ;
       FlightPath*          getFlightPath() const ;
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
#ifdef OPTICKS_NPYSERVER
       numpydelegate*              m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       BCfg*                m_cfg ;
       OpticksCfg<Opticks>* m_fcfg ;   
       NState*              m_state ; 
       NLookup*             m_lookup ; 
       Bookmarks*           m_bookmarks ; 
       FlightPath*          m_flightpath ; 

   private:
       OpticksGen*          m_gen ; 
       OpticksAim*          m_aim ;
 
       GGeoTest*            m_geotest ; 
       int                  m_err ; 
       SCtrl*               m_ctrl ; 



};


