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

/*
T* CProcessManger::GetProcess<T>( G4ProcessManager* proMgr, const char* processName_ )
{
    G4String processName = processName_ ; 
    G4VProcess* proc = proMgr->GetProcess(processName); 
    return proc ?  dynamic_cast<T*>(proc) : nullptr ; 
}
*/


G4ProcessManager* CProcessManager::Current(G4Track* trk) 
{
    const G4ParticleDefinition* defn = trk->GetDefinition() ;
    return defn->GetProcessManager() ;
}


/**
CProcessManager::ResetNumberOfInteractionLengthLeft
------------------------------------------------------

G4VProcess::ResetNumberOfInteractionLengthLeft explicity invokes G4UniformRand
whereas ClearNumberOfInteractionLengthLeft induces that to happen
for the next step

g4-;g4-cls G4VProcess::

    303  public: // with description
    304       virtual void      ResetNumberOfInteractionLengthLeft();
    305      // reset (determine the value of)NumberOfInteractionLengthLeft
    ...
    314  protected:  // with description
    ...
    321      void      ClearNumberOfInteractionLengthLeft();
    322      // clear NumberOfInteractionLengthLeft 
    323      // !!! This method should be at the end of PostStepDoIt()
    324      // !!! and AtRestDoIt
    ...
    450 inline void G4VProcess::ClearNumberOfInteractionLengthLeft()
    451 {
    452   theInitialNumberOfInteractionLength = -1.0;
    453   theNumberOfInteractionLengthLeft =  -1.0;
    454 }

    095 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     96 {
     97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     99 }

**/

void CProcessManager::ResetNumberOfInteractionLengthLeft(G4ProcessManager* proMgr)
{

    G4ProcessVector* pl = proMgr->GetProcessList() ;
    G4int n = pl->entries() ;

    for(int i=0 ; i < n ; i++)
    {
        G4VProcess* p = (*pl)[i] ; 
        p->ResetNumberOfInteractionLengthLeft();
    } 
}

/**
CProcessManager::ClearNumberOfInteractionLengthLeft
-----------------------------------------------------

This simply clears the interaction length left for OpAbsorption and OpRayleigh 
with no use of G4UniformRand.

This provides a devious way to invoke the protected ClearNumberOfInteractionLengthLeft 
via the public G4VDiscreteProcess::PostStepDoIt

g4-;g4-cls G4VDiscreteProcess::

    112 G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    113                             const G4Track& ,
    114                             const G4Step&
    115                             )
    116 { 
    117 //  clear NumberOfInteractionLengthLeft
    118     ClearNumberOfInteractionLengthLeft();
    119 
    120     return pParticleChange;
    121 }

**/

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


