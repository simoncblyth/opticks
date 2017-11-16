#pragma once

#include <string>
#include <map>

// g4-
class G4RunManager ; 
class G4VisManager ; 
class G4UImanager ; 
class G4UIExecutive ; 
class G4VUserDetectorConstruction ;

// npy-
template <typename T> class NPY ; 

// cfg4-


class CPhysics ; 
class CGeometry ; 
class CMaterialLib ; 
class CDetector ; 
class CMaterialBridge ; 
class CSurfaceBridge ; 
class CGenerator ; 

class CCollector ; 
class CRecorder ; 
class CStepRec ; 

//class ActionInitialization ;
class CRunAction ; 
class CEventAction ; 
class CPrimaryGeneratorAction ;
class CTrackingAction ; 
class CSteppingAction ; 

class OpticksHub ; 
class OpticksRun ; 
class OpticksEvent ; 
class Opticks ; 
template <typename T> class OpticksCfg ;

/**

CG4
====

Canonical instance m_g4 is resident of OKG4Mgr and is instanciated 
with it for non "--load" option running.

Prime method CG4::propagate is invoked from OKG4Mgr::propagate



Whats the difference between CRecorder/m_recorder and CStepRec/m_steprec ?
-------------------------------------------------------------------------------

CStepRec 
   records non-optical particle steps into the m_nopstep buffer of the
   OpticksEvent set by CStepRec::initEvent
    
CRecorder
   records optical photon steps and photon tracks


CStepRec is beautifully simple, CRecorder is horrible complicated in comparison


Workflow overview
-------------------

Traditional GPU Opticks simulation workflow:

* gensteps (Cerenkov/Scintillation) harvested from Geant4
  and persisted into OpticksEvent

* gensteps seeded onto GPU using Thrust, summation over photons 
  to generate per step provide photon and record buffer 
  dimensions up frount 

* Cerenkov/Scintillation on GPU generation and propagation      
  populate the pre-sized GPU record buffer 

This works because all gensteps are available before doing 
any optical simulation. BUT when operating on CPU doing the 
non-optical and optical simulation together, do not know the 
photon counts ahead of time.

**/


#include "CG4Ctx.hh"

#include "CFG4_API_EXPORT.hh"

class CFG4_API CG4 
{
        friend class CGeometry ; 
   public:
        CG4(OpticksHub* hub);
        void interactive();
        void cleanup();
        bool isDynamic(); // true for G4GUN without gensteps ahead of time, false for TORCH with gensteps ahead of time
   public:
        void initialize();
        NPY<float>* propagate();
   private:
        void postinitialize();
        void postpropagate();
   public:
        CEventAction*    getEventAction();
        CSteppingAction* getSteppingAction();
        CTrackingAction* getTrackingAction();
        //int getStepId();
   public:
        std::map<std::string, unsigned>& getMaterialMap();        
   private:
        void init();
        void setUserInitialization(G4VUserDetectorConstruction* detector);
        void execute(const char* path);
        void initEvent(OpticksEvent* evt);
   public:
        Opticks*       getOpticks();
        OpticksHub*    getHub();
        CGeometry*     getGeometry();
        CMaterialBridge* getMaterialBridge();
        CSurfaceBridge*  getSurfaceBridge();
        CG4Ctx&        getCtx();

        CRecorder*     getRecorder();
        CStepRec*      getStepRec();
        CMaterialLib*  getMaterialLib();
        CDetector*     getDetector();
        NPY<float>*    getGensteps();
   private:
        OpticksHub*           m_hub ; 
        Opticks*              m_ok ; 
        OpticksRun*           m_run ; 
        OpticksCfg<Opticks>*  m_cfg ; 

        CG4Ctx                m_ctx ;       

        CPhysics*             m_physics ; 
        G4RunManager*         m_runManager ; 
        CGeometry*            m_geometry ; 
        bool                  m_hookup ; 
        CMaterialLib*         m_mlib ; 
        CDetector*            m_detector ; 
        CGenerator*           m_generator ; 
        bool                  m_dynamic ; 
   private:
        CCollector*           m_collector ; 
        CRecorder*            m_recorder ; 
        CStepRec*             m_steprec ; 
   private:
        G4VisManager*         m_visManager ; 
        G4UImanager*          m_uiManager ; 
        G4UIExecutive*        m_ui ; 
   private:

        CPrimaryGeneratorAction*       m_pga ; 
        CSteppingAction*               m_sa ;
        CTrackingAction*               m_ta ;
        CRunAction*                    m_ra ;
        CEventAction*                  m_ea ;

        bool                           m_initialized ; 


        
};

