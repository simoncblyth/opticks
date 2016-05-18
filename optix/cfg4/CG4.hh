#pragma once
#include <cstdlib>

class G4RunManager ; 
class G4VisManager ; 
class G4UImanager ; 
class G4UIExecutive ; 
class G4VUserDetectorConstruction ;
class G4VUserPrimaryGeneratorAction ;
class G4UserSteppingAction ;

class CG4 
{
   public:
        CG4();
        void configure(int argc, char** argv);
        void interactive(int argc, char** argv);
        virtual ~CG4();
   public:
        void setDetectorConstruction(G4VUserDetectorConstruction* dc);
        void setPrimaryGeneratorAction(G4VUserPrimaryGeneratorAction* pga);
        void setSteppingAction(G4UserSteppingAction* sa);
        void initialize();
   public:
        void BeamOn(unsigned int num);
   private:
        G4RunManager*         m_runManager ;
   private:
        bool                  m_g4ui ; 
        G4VisManager*         m_visManager ; 
        G4UImanager*          m_uiManager ; 
        G4UIExecutive*        m_ui ; 
   private:
        G4VUserPrimaryGeneratorAction* m_pga ; 
        G4UserSteppingAction*          m_sa ; 
        
};

inline CG4::CG4() 
   :
     m_runManager(NULL),
     m_g4ui(false),
     m_visManager(NULL),
     m_uiManager(NULL),
     m_ui(NULL),
     m_pga(NULL),
     m_sa(NULL)
{
}

inline void CG4::setPrimaryGeneratorAction(G4VUserPrimaryGeneratorAction* pga)
{
    m_pga = pga ; 
}
inline void CG4::setSteppingAction(G4UserSteppingAction* sa)
{
    m_sa = sa ; 
}




