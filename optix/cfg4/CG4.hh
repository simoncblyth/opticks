#pragma once
#include <cstdlib>

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

//#define OLDPHYS 1
#ifdef OLDPHYS
class PhysicsList ; 
#else
class OpNovicePhysicsList ; 
#endif

#include "OpticksEngine.hh"

class CG4 : public OpticksEngine
{
   public:
        CG4(Opticks* opticks);
        void configure();
        void interactive(int argc, char** argv);
        void cleanup();
        virtual ~CG4();
   public:
        void initialize();
        void propagate();
        void save();
   private:
        void init();
        //void initEvent();
        void configureDetector();
        void configurePhysics();
        void configureGenerator();
        void configureStepping();
   private:
        void postinitialize();
        void postpropagate();
        void setupCompressionDomains();
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

inline CG4::CG4(Opticks* opticks) 
   :
     OpticksEngine(opticks),
     m_torch(NULL),
     m_detector(NULL),
     m_lib(NULL),
     m_recorder(NULL),
     m_rec(NULL),
     m_steprec(NULL),
     m_physics(NULL),
     m_runManager(NULL),
     m_g4ui(false),
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(NULL),
     m_sa(NULL)
{
     init();
}


inline CRecorder* CG4::getRecorder()
{
    return m_recorder ; 
}
inline Rec* CG4::getRec()
{
    return m_rec ; 
}
inline CStepRec* CG4::getStepRec()
{
    return m_steprec ; 
}
inline CPropLib* CG4::getPropLib()
{
    return m_lib ; 
}


