
#include <sstream>
#include <iomanip>

#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4VDiscreteProcess.hh"

#include "CProcessManager.hh"


std::string CProcessManager::Desc( G4ProcessManager* proMgr )
{
    std::stringstream ss ; 
    ss << "CProMgr" ;

    G4ProcessVector* pl = proMgr->GetProcessList() ;
    G4int n = pl->entries() ;

    ss << " n:[" << n << "]" ;  
    for(int i=0 ; i < n ; i++)
    {
        G4VProcess* p = (*pl)[i] ; 
        ss << " (" << i << ")" 
           << " name " << p->GetProcessName() 
           << " left " << p->GetNumberOfInteractionLengthLeft()
           ; 
    } 
    return ss.str();
}


G4ProcessManager* CProcessManager::Current(G4Track* trk) 
{
    const G4ParticleDefinition* defn = trk->GetDefinition() ;
    return defn->GetProcessManager() ;
}




void CProcessManager::ResetNumberOfInteractionLengthLeft(G4ProcessManager* proMgr)
{
    // ResetNumberOfInteractionLengthLeft explicity throws the G4UniformRand
    // whereas ClearNumberOfInteractionLengthLeft should induce that to happen
    // for next step

    G4ProcessVector* pl = proMgr->GetProcessList() ;
    G4int n = pl->entries() ;

    for(int i=0 ; i < n ; i++)
    {
        G4VProcess* p = (*pl)[i] ; 
        p->ResetNumberOfInteractionLengthLeft();
    } 
}
/*

     95 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     96 {
     97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     99 }

*/



void CProcessManager::ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep)
{
    G4ProcessVector* pl = proMgr->GetProcessList() ;
    G4int n = pl->entries() ;

    for(int i=0 ; i < n ; i++)
    {
        G4VProcess* p = (*pl)[i] ; 
        const G4String& name = p->GetProcessName() ;
        bool is_ab = name.compare("OpAbsorption") == 0 ;
        bool is_sc = name.compare("OpRayleigh") == 0 ;
        //bool is_bd = name.compare("OpBoundary") == 0 ;
        if( is_ab || is_sc )
        {
            G4VDiscreteProcess* dp = dynamic_cast<G4VDiscreteProcess*>(p) ;     
            assert(dp);   // Transportation not discrete
            dp->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );   
            // devious way to invoke the protected ClearNumberOfInteractionLengthLeft via G4VDiscreteProcess::PostStepDoIt
        }
    } 
}


G4VDiscreteProcess* CProcessManager::GetDiscreteProcess( G4ProcessManager* proMgr, const char* name)
{
    G4ProcessVector* pl = proMgr->GetProcessList() ;
    G4int n = pl->entries() ;
    G4VProcess* p = NULL ; 
    for(int i=0 ; i < n ; i++)
    {
        p = (*pl)[i] ; 
        const G4String& pname = p->GetProcessName() ;
        if(pname.compare(name) == 0) break ; 
    }
      
    G4VDiscreteProcess* dp = dynamic_cast<G4VDiscreteProcess*>(p) ;
    return dp ; 
}



void CProcessManager::ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep, const char* name)
{
    G4VDiscreteProcess* dp = GetDiscreteProcess(proMgr, name); 
    assert(dp);
    dp->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
    //   devious way to invoke the protected ClearNumberOfInteractionLengthLeft 
}




