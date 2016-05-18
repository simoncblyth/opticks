
//g4-
#include "G4RunManager.hh"
#include "G4String.hh"

#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"

//cg4-
#include "CG4.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"


void CG4::configure(int argc, char** argv)
{
    m_runManager = new G4RunManager;
    m_runManager->SetUserInitialization(new PhysicsList());
    m_g4ui = false ; 
}

void CG4::setDetectorConstruction(G4VUserDetectorConstruction* dc)
{
    m_runManager->SetUserInitialization(dc);
}

void CG4::initialize()
{
    m_runManager->SetUserInitialization(new ActionInitialization(m_pga, m_sa)) ;
    m_runManager->Initialize();
}



void CG4::interactive(int argc, char** argv)
{
    if(!m_g4ui) return ; 

    m_visManager = new G4VisExecutive;
    m_visManager->Initialize();

    m_uiManager = G4UImanager::GetUIpointer();

    m_ui = new G4UIExecutive(argc, argv);

    m_ui->SessionStart();
}

void CG4::BeamOn(unsigned int num)
{
    m_runManager->BeamOn(num);
}


CG4::~CG4()
{
    delete m_runManager;
}



