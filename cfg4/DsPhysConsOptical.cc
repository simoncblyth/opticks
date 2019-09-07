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

#define USE_CUSTOM_CERENKOV
#define USE_CUSTOM_SCINTILLATION

#include "DsPhysConsOptical.h"
#include "DsG4OpRayleigh.h"

#ifdef USE_CUSTOM_CERENKOV
#include "DsG4Cerenkov.h"
#else
#include "G4Cerenkov.hh"
#endif

#ifdef USE_CUSTOM_SCINTILLATION
#include "DsG4Scintillation.h"
#else
#include "G4Scintillation.hh"
#endif

#include "PLOG.hh"


#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
//#include "G4OpBoundaryProcess.hh"
#include "DsG4OpBoundaryProcess.h"
#include "G4ProcessManager.hh"
#include "G4FastSimulationManagerProcess.hh"

using CLHEP::g ; 
using CLHEP::cm2 ; 
using CLHEP::MeV ; 
using CLHEP::ns ; 



DsPhysConsOptical::DsPhysConsOptical(Opticks* ok)
   :
      m_ok(ok),
      m_doReemission(true),               // "ScintDoReemission"        "Do reemission in scintilator."
      m_doScintAndCeren(true),            // "ScintDoScintAndCeren"     "Do both scintillation and Cerenkov in scintilator."
      m_useFastMu300nsTrick(false),       // "UseFastMu300nsTrick"      "Use Fast muon simulation?"
      m_useCerenkov(true),                // "UseCerenkov"              "Use the Cerenkov process?"
      m_useScintillation(true),           // "UseScintillation"         "Use the Scintillation process?"
      m_useRayleigh(true),                // "UseRayleigh"              "Use the Rayleigh scattering process?"
      m_useAbsorption(true),              // "UseAbsorption"            "Use light absorption process?"
      m_applyWaterQe(false),              // "ApplyWaterQe"             
                                          // "Apply QE for water cerenkov process when OP is created? 
                                          // If it is true the CerenPhotonScaleWeight will be disabled in water, but it still works for AD and others "
      m_cerenPhotonScaleWeight(3.125),    // "CerenPhotonScaleWeight"    "Scale down number of produced Cerenkov photons by this much."
      m_cerenMaxPhotonPerStep(300),       // "CerenMaxPhotonsPerStep"   "Limit step to at most this many (unscaled) Cerenkov photons."
      m_scintPhotonScaleWeight(3.125),    // "ScintPhotonScaleWeight"    "Scale down number of produced scintillation photons by this much."
      m_ScintillationYieldFactor(1.0),    // "ScintillationYieldFactor" "Scale the number of scintillation photons per MeV by this much."
      m_birksConstant1(6.5e-3*g/cm2/MeV), // "BirksConstant1"           "Birks constant C1"
      m_birksConstant2(3.0e-6*(g/cm2/MeV)*(g/cm2/MeV)),  
                                          // "BirksConstant2"            "Birks constant C2" 
      m_gammaSlowerTime(149*ns),          // "GammaSlowerTime"           "Gamma Slower time constant"
      m_gammaSlowerRatio(0.338),          // "GammaSlowerRatio"          "Gamma Slower time ratio"
      m_neutronSlowerTime(220*ns),        // "NeutronSlowerTime"         "Neutron Slower time constant"
      m_neutronSlowerRatio(0.34),         // "NeutronSlowerRatio"        "Neutron Slower time ratio"
      m_alphaSlowerTime(220*ns),          // "AlphaSlowerTime"           "Alpha Slower time constant"
      m_alphaSlowerRatio(0.35)            // "AlphaSlowerRatio"          "Alpha Slower time ratio"
{
}

void DsPhysConsOptical::dump(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info)<<"Photons prescaling is "<<( m_cerenPhotonScaleWeight>1.?"on":"off" )
             <<" for Cerenkov. Preliminary applied efficiency is "
             <<1./m_cerenPhotonScaleWeight<<" (weight="<<m_cerenPhotonScaleWeight<<")" ;
    LOG(info)<<"Photons prescaling is "<<( m_scintPhotonScaleWeight>1.?"on":"off" )
             <<" for Scintillation. Preliminary applied efficiency is "
             <<1./m_scintPhotonScaleWeight<<" (weight="<<m_scintPhotonScaleWeight<<")";
    LOG(info)<<"WaterQE is turned "<<(m_applyWaterQe?"on":"off")<<" for Cerenkov.";
}


void DsPhysConsOptical::ConstructProcess()
{

    assert(0); 

#ifdef USE_CUSTOM_CERENKOV
    
    LOG(info)  << "Using customized DsG4Cerenkov." ;
    DsG4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) 
    {
        cerenkov = new DsG4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetApplyPreQE(m_cerenPhotonScaleWeight>1.);
        cerenkov->SetPreQE(1./m_cerenPhotonScaleWeight);
        cerenkov->SetApplyWaterQe(m_applyWaterQe);
        cerenkov->SetTrackSecondariesFirst(true);
    }
#else
    LOG(info) << "Using standard G4Cerenkov." ;
    G4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) 
    {
        cerenkov = new G4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetTrackSecondariesFirst(true);
    }
#endif

#ifdef USE_CUSTOM_SCINTILLATION
    DsG4Scintillation* scint = 0;
    LOG(info) << "Using customized DsG4Scintillation." ;
    scint = new DsG4Scintillation();
    scint->SetBirksConstant1(m_birksConstant1);
    scint->SetBirksConstant2(m_birksConstant2);
    scint->SetGammaSlowerTimeConstant(m_gammaSlowerTime);
    scint->SetGammaSlowerRatio(m_gammaSlowerRatio);
    scint->SetNeutronSlowerTimeConstant(m_neutronSlowerTime);
    scint->SetNeutronSlowerRatio(m_neutronSlowerRatio);
    scint->SetAlphaSlowerTimeConstant(m_alphaSlowerTime);
    scint->SetAlphaSlowerRatio(m_alphaSlowerRatio);
    scint->SetDoReemission(m_doReemission);
    scint->SetDoBothProcess(m_doScintAndCeren);
    scint->SetApplyPreQE(m_scintPhotonScaleWeight>1.);
    scint->SetPreQE(1./m_scintPhotonScaleWeight);
    scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); //1.);
    scint->SetUseFastMu300nsTrick(m_useFastMu300nsTrick);
    scint->SetTrackSecondariesFirst(true);
    if (!m_useScintillation) scint->SetNoOp();
#else  // standard G4 scint
    G4Scintillation* scint = 0;
    if (m_useScintillation) 
    {
        LOG(info) << "Using standard G4Scintillation." ;
        scint = new G4Scintillation();
        scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); // 1.);
        scint->SetTrackSecondariesFirst(true);
    }
#endif

    G4OpAbsorption* absorb  = m_useAbsorption ? new G4OpAbsorption() : NULL ;
    DsG4OpRayleigh* rayleigh = m_useRayleigh  ? new DsG4OpRayleigh() : NULL ; 

    //G4OpBoundaryProcess* boundproc = new G4OpBoundaryProcess();
    DsG4OpBoundaryProcess* boundproc = new DsG4OpBoundaryProcess();
    boundproc->SetModel(unified);

    //G4FastSimulationManagerProcess* fast_sim_man = new G4FastSimulationManagerProcess("fast_sim_man");
    
    theParticleIterator->reset();
    while( (*theParticleIterator)() ) {

        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
    
        // Caution: as of G4.9, Cerenkov becomes a Discrete Process.
        // This code assumes a version of G4Cerenkov from before this version.
        //
        /// SCB: Contrary to above FUD-comment, contemporary G4 code such as 
        ///      OpNovicePhysicsList sets up Cerenkov just like this

        if(cerenkov && cerenkov->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(cerenkov);
            pmanager->SetProcessOrdering(cerenkov, idxPostStep);
            LOG(debug) << "Process: adding Cherenkov to " 
                       << particle->GetParticleName() ;
        }

/*
        if(scint && scint->IsApplicable(*particle))
        {
            pmanager->AddProcess(scint);
            pmanager->SetProcessOrderingToLast(scint, idxAtRest);
            pmanager->SetProcessOrderingToLast(scint, idxPostStep);
            LOG(debug) << "Process: adding Scintillation to "
                       << particle->GetParticleName() ;
        }
*/

        if(particle == G4OpticalPhoton::Definition()) 
        {
            if(absorb) pmanager->AddDiscreteProcess(absorb);
            if(rayleigh) pmanager->AddDiscreteProcess(rayleigh);
            pmanager->AddDiscreteProcess(boundproc);
            //pmanager->AddDiscreteProcess(fast_sim_man);
        }
    }
}




