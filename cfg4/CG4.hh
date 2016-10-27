#pragma once

#include <string>
#include <map>

// g4-
class G4RunManager ; 
class G4VisManager ; 
class G4UImanager ; 
class G4UIExecutive ; 
class G4VUserDetectorConstruction ;
class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;
class G4UserRunAction ;
class G4UserEventAction ;

// npy-
template <typename T> class NPY ; 

// cfg4-
class CPhysics ; 
class CGeometry ; 
class CPropLib ; 
class CDetector ; 
class CMaterialBridge ; 
class CGenerator ; 

class CCollector ; 
class CRecorder ; 
//class Rec ; 
class CStepRec ; 

class OpticksHub ; 
class OpticksRun ; 
class OpticksEvent ; 
class Opticks ; 
template <typename T> class OpticksCfg ;


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
        std::map<std::string, unsigned>& getMaterialMap();        
   private:
        void init();
        void setUserInitialization(G4VUserDetectorConstruction* detector);
        void execute(const char* path);
        void initEvent(OpticksEvent* evt);
   public:
        Opticks*       getOpticks();
        CGeometry*     getGeometry();
        CMaterialBridge* getMaterialBridge();

        CRecorder*     getRecorder();
        CStepRec*      getStepRec();
        CPropLib*      getPropLib();
        CDetector*     getDetector();
        NPY<float>*    getGensteps();
   private:
        OpticksHub*           m_hub ; 
        Opticks*              m_ok ; 
        OpticksRun*           m_run ; 
        OpticksCfg<Opticks>*  m_cfg ; 
        CPhysics*             m_physics ; 
        G4RunManager*         m_runManager ; 
        CGeometry*            m_geometry ; 
        bool                  m_hookup ; 
        CPropLib*             m_lib ; 
        CDetector*            m_detector ; 
        CGenerator*           m_generator ; 
   private:
        CCollector*           m_collector ; 
        CRecorder*            m_recorder ; 
        CStepRec*             m_steprec ; 
   private:
        G4VisManager*         m_visManager ; 
        G4UImanager*          m_uiManager ; 
        G4UIExecutive*        m_ui ; 
   private:
        G4VUserPrimaryGeneratorAction* m_pga ; 
        G4UserSteppingAction*          m_sa ; 
        G4UserRunAction*               m_ra ; 
        G4UserEventAction*             m_ea ; 
        bool                           m_initialized ; 
        
};

