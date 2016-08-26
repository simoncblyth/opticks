#pragma once

// g4-
class G4RunManager ; 
class G4VisManager ; 
class G4UImanager ; 
class G4UIExecutive ; 
class G4VUserDetectorConstruction ;
class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;

// npy--
class TorchStepNPY ;

// cfg4-
class CPropLib ; 
class CDetector ; 
class CRecorder ; 
class Rec ; 
class CStepRec ; 

class OpticksHub ; 

//#define OLDPHYS 1
#ifdef OLDPHYS
class PhysicsList ; 
#else
class OpNovicePhysicsList ; 
#endif

#include "OpticksEngine.hh"
#include "CFG4_API_EXPORT.hh"

class CFG4_API CG4 : public OpticksEngine
{
   public:
        CG4(OpticksHub* hub);
        void configure();
        void interactive(int argc, char** argv);
        void cleanup();
   public:
        void initialize();
        void propagate();
   private:
        void init();
        void configureDetector();
        void configurePhysics();
        void configureGenerator();
        void configureStepping();
   private:
        void postinitialize();
        void postpropagate();
   private:
        void execute(const char* path);
   public:
        void BeamOn(unsigned int num);
   public:
        CRecorder* getRecorder();
        CStepRec* getStepRec();
        Rec*      getRec();
        CPropLib* getPropLib();
   private:
        TorchStepNPY*         m_torch ; 
   private:
        CDetector*            m_detector ; 
        CPropLib*             m_lib ; 
        CRecorder*            m_recorder ; 
        Rec*                  m_rec ; 
        CStepRec*             m_steprec ; 
   private:
#ifdef OLDPHYS
        PhysicsList*          m_physics ; 
#else
        OpNovicePhysicsList*  m_physics ; 
#endif
        G4RunManager*         m_runManager ;
        bool                  m_g4ui ; 
        G4VisManager*         m_visManager ; 
        G4UImanager*          m_uiManager ; 
        G4UIExecutive*        m_ui ; 
   private:
        G4VUserPrimaryGeneratorAction* m_pga ; 
        G4UserSteppingAction*          m_sa ; 
        
};

