/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <string>
#include <map>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

class SCtrl ; 
class BCfg ; 

class Opticks ; 
class OpticksAttrSeq ; 
class OpticksEvent ; 

class GenstepNPY  ; 

class GGeoBase ; 
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

TODO: reduce the number of code paths, perhaps with a GGeo argument to OpticksHub


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
       static int Preinit(); 
       void init();
       void setErr(int err);
       void configure();
       void configureCompositionSize();

       void loadGeometry();
       void adoptGeometry();

   private:
       void configureServer();
#ifdef LEGACY
       void configureLookupA();
#endif
       void overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg);
       void overrideMaterialMapA(const char* jsonA );
       void setupTestGeometry(); 
   public:
       void add(BCfg* cfg);
       BCfg* getUmbrellaCfg() const ; 
   public:
       bool         hasOpt(const char* name);
       bool         isCompute();
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

   public:
       GGeo*                getGGeo() const ;
       GGeoBase*            getGGeoBase() const ; // downcast of the encumbent: GGeoTest/GScene/GGeo 

   public:
       //glm::mat4            getTransform(int index);

   private:

       GGeoBase*            getGGeoBasePrimary() const ;  //
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

     //  GPmtLib*             getPmtLib();       //   partlist? analytic PMT   
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
       OpticksAttrSeq*      getFlagNames();

       std::string          desc() const ;
   public:
       OpticksGen*          getGen();
   public:
       void configureVizState(NConfigurable* scene);
       void setupFlightPath();
       void cleanup();

   private:
       int              m_preinit ; 
       Opticks*         m_ok ; 
       int              m_gltf ;
       bool             m_immediate ; 

       GGeo*            m_ggeo ;  
       Composition*     m_composition ; 

   private:
#ifdef OPTICKS_NPYSERVER
       numpydelegate*              m_delegate ; 
       numpyserver<numpydelegate>* m_server ;
#endif
       BCfg*                m_umbrella_cfg ;
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


