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

#include "CFG4_BODY.hh"
#include <sstream>


#include "CFG4_PUSH.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "CFG4_POP.hh"


#include "Format.hh"
#include "CStep.hh"
#include "OpStatus.hh"


std::string Format(const G4double v, const char* msg, unsigned int fwid)
{
    std::stringstream ss ; 
    ss << " " << msg << "[ "
       << std::fixed
       << std::setprecision(3)  
       << std::setw(fwid)
       << v 
       ;
    return ss.str();
}

std::string Format(const G4ThreeVector& vec, const char* msg, unsigned int fwid)
{
    std::stringstream ss ; 
    ss << " " << msg << "[ "
       << std::fixed
       << std::setprecision(3)  
       << std::setw(fwid) << vec.x()
       << std::setw(fwid) << vec.y() 
       << std::setw(fwid) << vec.z() 
       << " ; "
       << std::setw(fwid) << vec.mag() 
       << "] "
       ;
    return ss.str();
}


std::string Format(const char* label, std::string pre, std::string post, unsigned int w)
{
    std::stringstream ss ; 

    ss 
       << " " 
       << std::setw(w) << label 
       << " ["
       << std::setw(w) << pre
       << "/" 
       << std::setw(w) << post
       << "]" 
       ;

    return ss.str();
}



std::string Format(const G4Track* track, const G4ThreeVector& origin, const char* msg, bool op)
{
    G4int tid = track->GetTrackID();
    G4int pid = track->GetParentID();

    G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

    const G4ThreeVector& pos = track->GetPosition();
    G4double energy = track->GetKineticEnergy() ;
    G4double wavelength= h_Planck*c_light/energy ;

    G4ThreeVector offpos = pos - origin ;  

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << particleName 
       << " tid " << tid
       << " pid " << pid
       << ( op ? " nm " : " keV " )
       << std::setw(6) << ( op ? wavelength/nm : energy/keV ) 
       << " mm "
       << Format(origin, "ori", 8)
       << Format(offpos, "pos", 8)
       << " )"
       ;
    return ss.str();
}

std::string Tail(const G4String& _str, unsigned int n )
{
    std::string str = _str ; 
    std::string ss = str.size() < n ? str : str.substr(str.size() - n );
    return ss ; 
}


std::string Format(const G4StepPoint* point, const G4ThreeVector& origin, const char* msg, bool op )
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4ThreeVector offpos = pos - origin ;  

    const G4String noMaterial = "noMaterial" ; 
    const G4String noProc = "noProc" ; 

    const G4Material* mat = point->GetMaterial() ;
    const G4String& matName = mat ? mat->GetName() : noMaterial ; 

    G4VPhysicalVolume* pv  = point->GetPhysicalVolume();
    G4String pvName = pv ? pv->GetName() : "" ;   

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double velocity = point->GetVelocity();


    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : noProc ; 

    G4StepStatus status = point->GetStepStatus()  ;

    std::stringstream ss ; 
    ss << " " << std::setw(4)  << msg  
       << " " << std::setw(25) << Tail(pvName, 25)  
       << " " << std::setw(15) << Tail(matName, 15) 
       << " " << std::setw(15) << Tail(processName, 15) 
       << std::setw(20) << OpStatus::OpStepString(status)
       << Format(offpos, "pos", 10)
       << Format(dir, "dir", 8)
       << Format(pol, "pol", 8)
       << std::setprecision(3) << std::fixed
       << " ns " << std::setw(6) << time/ns 
       << ( op ? " nm " : " keV " ) 
       << std::setw(6) << ( op ? wavelength/nm : energy/keV )
       << " mm/ns "
       << std::setw(6) << velocity
       ;

    return ss.str();
}


std::string Format(const G4StepPoint* pre, const G4StepPoint* post, double epsilon, const char* msg )
{
    const G4ThreeVector& pre_pos = pre->GetPosition();
    const G4ThreeVector& pre_dir = pre->GetMomentumDirection();
    const G4ThreeVector& pre_pol = pre->GetPolarization();
    G4double pre_time = pre->GetGlobalTime();

    const G4ThreeVector& post_pos = post->GetPosition();
    const G4ThreeVector& post_dir = post->GetMomentumDirection();
    const G4ThreeVector& post_pol = post->GetPolarization();
    G4double post_time = post->GetGlobalTime();

    bool same_pos = pre_pos == post_pos ; 
    bool same_dir = pre_dir == post_dir ; 
    bool same_pol = pre_pol == post_pol ; 
    bool same_time = pre_time == post_time ; 

    bool near_pos = pre_pos.isNear(post_pos, epsilon) ; 
    bool near_dir = pre_dir.isNear(post_dir, epsilon) ; 
    bool near_pol = pre_pol.isNear(post_pol, epsilon) ; 
    bool near_time = std::abs( pre_time - post_time ) < epsilon ; 

    G4ThreeVector dpos = post_pos - pre_pos ; 
    G4ThreeVector ddir = post_dir - pre_dir ; 
    G4ThreeVector dpol = post_pol - pre_pol ; 
    double dtim = post_time - pre_time ; 

    std::stringstream ss ; 
    ss << " " << std::setw(4)  << msg  ;


    ss << Format(dpos, "dpos", 8) ; 
    if( same_pos )
    {
       ss << " " << "same_pos"  ;
    }
    else if( near_pos )
    {
       ss << " " << "near_pos" ; 
    }


    ss << Format(ddir, "ddir", 8) ; 
    if( same_dir )
    {
       ss << " " << "same_dir"  ;
    }
    else if( near_dir )
    {
       ss << " " << "near_dir" ; 
    }


    ss << Format(dpol, "dpol", 8) ;
    if( same_pol )
    {
       ss << " " << "same_pol"  ;
    }
    else if( near_pol )
    {
       ss << " " << "near_pol" ; 
    }


    ss << Format(dtim, "dtim", 8) ;
    if( same_time )
    {
       ss << " " << "same_time"  ;
    }
    else if( near_time )
    {
       ss << " " << "near_time" ; 
    }



    ss << "       epsilon " << epsilon ; 
    return ss.str();
}



std::string Format(const G4Step* step, const G4ThreeVector& origin, const char* msg, bool op )
{
    G4Track* track = step->GetTrack();
    G4int stepNum = track->GetCurrentStepNumber() ;
    //  gives crazy stepNum from persisted step, persumably depending on live not persisted in Step state
    G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

    G4StepPoint* pre  = step->GetPreStepPoint() ;
    G4StepPoint* post = step->GetPostStepPoint() ;

    std::stringstream ss ; 
    ss << "(" << msg << " ;" 
       << particleName 
       << " stepNum " << std::setw(4) << stepNum  
       << Format(track, origin,  "tk", op) << "\n"
       << Format(pre,  origin, "pre", op) << "\n"
       << Format(post, origin, "post", op) << "\n"
       << Format(pre, post ) << "\n"
       << " )"
       ;

    return ss.str();
}


std::string Format(std::vector<const CStep*>& steps, const char* msg, bool op)
{
    unsigned int nsteps = steps.size();

    std::stringstream ss ; 

    G4ThreeVector origin ;

    for(unsigned int i=0 ; i < nsteps ; i++)
    {
         const CStep* cstep = steps[i] ;
         const G4Step* step = cstep->getStep();

         //unsigned int step_id = cstep->getStepId();
        //  assert(step_id == i );  not always so ?

         G4StepPoint* pre = step->GetPreStepPoint() ;

         if(i == 0)
         {
             G4Track* track = step->GetTrack();
             G4String particleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName();

             const G4ThreeVector& pos = pre->GetPosition();
             origin = pos ; 

             ss << "(" << msg << " ;" 
                << particleName 
                << " nsteps " << std::setw(4) << nsteps
                << Format(track, origin, "tk", op) << "\n"
                ;
         }

         ss << Format(pre, origin, "pre", op) << "\n" ;

         if( i == nsteps - 1) 
         {
             G4StepPoint* post = step->GetPostStepPoint() ;
             ss << Format(post, origin, "post", op ) << "\n" ;
         }
    }
    return ss.str();
}






