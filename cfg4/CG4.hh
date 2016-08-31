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

// npy--
template <typename T> class NPY ; 
class TorchStepNPY ;

// cfg4-
class CPropLib ; 
class CDetector ; 
class CMaterialTable ; 
class CRecorder ; 
class Rec ; 
class CStepRec ; 
class OpticksG4Collector ; 

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
   public:
        std::map<std::string, unsigned>& getMaterialMap();        
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
        NPY<float>*   getGensteps();
   private:
        TorchStepNPY*         m_torch ; 
   private:
        CDetector*            m_detector ; 
        CMaterialTable*       m_material_table ; 
        CPropLib*             m_lib ; 
        CRecorder*            m_recorder ; 
        Rec*                  m_rec ; 
        CStepRec*             m_steprec ; 
        OpticksG4Collector*   m_collector ; 
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
        G4UserRunAction*               m_ra ; 
        G4UserEventAction*             m_ea ; 
        
};

