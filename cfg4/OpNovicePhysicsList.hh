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

#pragma once

#include <vector>
#include <set>

#include "globals.hh"
#include "G4VUserPhysicsList.hh"

class OpNovicePhysicsListMessenger;
class CG4 ; 
class Opticks ; 


// USE_CUSTOM_ 
#include "CProcessSwitches.hh"

class G4OpAbsorption;
class OpRayleigh;
class G4OpMieHG;

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
OpNovicePhysicsList
=====================

* this is messy : but important so make a new CPhysicsList implementation

Ideas:

* split off into CCerenkov and CScintillation
  where they do runtime implementation switch (not processor defines)
  to the different possible implementations
   

**/


class CFG4_API OpNovicePhysicsList : public G4VUserPhysicsList
{
  public:

    OpNovicePhysicsList(CG4* g4);
    virtual ~OpNovicePhysicsList();
  public:
    void Summary(const char* msg="OpNovicePhysicsList::Summary");
    void collectProcesses();
    void setProcessVerbosity(int verbosity);
    void setupEmVerbosity(unsigned int verbosity);
  public:
    void dump(const char* msg="OpNovicePhysicsList::dump");
  private:
    void dumpParam(const char* msg="OpNovicePhysicsList::dumpParam");
    void dumpRayleigh(const char* msg="OpNovicePhysicsList::dumpRayleigh");
    void dumpMaterials(const char* msg="OpNovicePhysicsList::dumpMaterials");
    void dumpProcesses(const char* msg="OpNovicePhysicsList::dumpProcesses");
    void addTransportation();
  public:

    
    virtual void ConstructParticle();  // method invoked by the RunManger, see g4-;g4-cls G4RunManagerKernel
    virtual void ConstructProcess();  

    virtual void SetCuts();

    //these methods Construct physics processes and register them
    void ConstructDecay();
    void ConstructEM();

    void ConstructOpDYB();   // alternate optical physics + scintillation and cerenkov
    void ConstructOpNovice();

    //for the Messenger 
    void SetVerbose(G4int);
    void SetNbOfPhotonsCerenkov(G4int);
 
  private:
    CG4*               m_g4 ;  
    Opticks*           m_ok ;  

    std::set<G4VProcess*> m_procs ; 
    std::vector<G4VProcess*> m_procl ; 

    OpNovicePhysicsListMessenger* fMessenger;

    static G4ThreadLocal G4int fVerboseLevel;
    static G4ThreadLocal G4int fMaxNumPhotonStep;

#ifdef USE_CUSTOM_CERENKOV
#  ifndef USE_CUSTOM_WITHGENSTEP_CERENKOV
    static G4ThreadLocal DsG4Cerenkov* fCerenkovProcess;
#  else
    static G4ThreadLocal Cerenkov* fCerenkovProcess;
#  endif
#else
    static G4ThreadLocal G4Cerenkov* fCerenkovProcess;
#endif

#ifdef USE_CUSTOM_SCINTILLATION
#  ifndef USE_CUSTOM_WITHGENSTEP_SCINTILLATION
    static G4ThreadLocal DsG4Scintillation* fScintillationProcess;
#  else
    static G4ThreadLocal Scintillation* fScintillationProcess;
#  endif
#else
    static G4ThreadLocal G4Scintillation* fScintillationProcess;
#endif

#ifdef USE_CUSTOM_BOUNDARY
    static G4ThreadLocal DsG4OpBoundaryProcess* fBoundaryProcess;
#else
    static G4ThreadLocal G4OpBoundaryProcess* fBoundaryProcess;
#endif

    static G4ThreadLocal G4OpAbsorption* fAbsorptionProcess;


#ifdef USE_POWER_THIRD_RAYLEIGH
    static G4ThreadLocal DsG4OpRayleigh* fRayleighScatteringProcess;
#else
    static G4ThreadLocal OpRayleigh* fRayleighScatteringProcess;
#endif

#ifdef USE_DEBUG_TRANSPORTATION
    static G4ThreadLocal DebugG4Transportation* fTransportationProcess;
#else
    static G4ThreadLocal G4Transportation* fTransportationProcess;
#endif




    static G4ThreadLocal G4OpMieHG* fMieHGScatteringProcess;


  private:
    // below adapted from DsPhysConsOptical

    bool m_doReemission;              /// ScintDoReemission: Do reemission in scintilator
    bool m_doScintAndCeren;           /// ScintDoScintAndCeren: Do both scintillation and Cerenkov in scintilator
    bool m_useFastMu300nsTrick; 
    bool m_useCerenkov ;
    bool m_useScintillation ;
    bool m_useRayleigh ;
    bool m_useAbsorption;
    bool m_applyWaterQe;              /// wangzhe: Apply QE for water cerenkov process when OP is created?  //     See DsG4Cerenkov and Doc 3925 for details
    double m_cerenPhotonScaleWeight;  /// Number (>= 1.0) used to scale down the mean number of optical photons produced.  
                                      /// For each actual secondary optical photon track produced, it will be given a weight equal to this scale
                                      /// for scaling up if detected later.  Default is 1.0.

    int m_cerenMaxPhotonPerStep;      /// Maximum number of photons per step to limit step size.  This value is independent from PhotonScaleWeight.  Default is 300.

    double m_scintPhotonScaleWeight;    /// Scale down number of produced scintillation photons by this much
    double m_ScintillationYieldFactor;  /// scale the number of produced scintillation photons per MeV by this much.
                                        /// This controls the primary yield of scintillation photons per MeV of deposited energy.
    double m_birksConstant1;           /// Birks constants C1 and C2
    double m_birksConstant2;
    double m_gammaSlowerTime;
    double m_gammaSlowerRatio;
    double m_neutronSlowerTime;
    double m_neutronSlowerRatio;
    double m_alphaSlowerTime;
    double m_alphaSlowerRatio;

};

#include "CFG4_TAIL.hh"


