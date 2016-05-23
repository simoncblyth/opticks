#pragma once
#include <cstdlib>

// optickscore-
class Opticks ; 
template <typename T> class OpticksCfg ;

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
class NumpyEvt ; 

// cfg4-
class CPropLib ; 
class CDetector ; 
class Recorder ; 
class Rec ; 
class OpNovicePhysicsList ; 

// ggeo-
class GCache ; 

class CG4 
{
   public:
        CG4(Opticks* opticks);
        void configure(int argc, char** argv);
        void interactive(int argc, char** argv);
        virtual ~CG4();
   public:
        void initialize();
        void propagate();
        void save();
   private:
        void init();
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
   private:
        Opticks*              m_opticks ; 
        OpticksCfg<Opticks>*  m_cfg ;
        GCache*               m_cache ; 
        NumpyEvt*             m_evt ; 
        TorchStepNPY*         m_torch ; 
   private:
        CDetector*            m_detector ; 
        CPropLib*             m_lib ; 
        Recorder*             m_recorder ; 
        Rec*                  m_rec ; 
   private:
        OpNovicePhysicsList*  m_npl ; 
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
     m_opticks(opticks),
     m_cfg(NULL),
     m_cache(NULL),
     m_evt(NULL),
     m_torch(NULL),
     m_detector(NULL),
     m_lib(NULL),
     m_recorder(NULL),
     m_rec(NULL),
     m_npl(NULL),
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




