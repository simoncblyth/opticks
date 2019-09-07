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

#include "DebugG4Transportation.hh"
#include "CG4.hh"
#include "CG4Ctx.hh"
#include "CTrack.hh"
#include "CMaterialLib.hh"
#include "Format.hh"

#include "PLOG.hh"

DebugG4Transportation::DebugG4Transportation( CG4* g4, G4int verbosity )
   :
    G4Transportation(verbosity),
    m_g4(g4),
    m_ctx(g4->getCtx()),
    m_mlib(g4->getMaterialLib()),
    m_origin(0,0,0)
{
}



G4VParticleChange* DebugG4Transportation::AlongStepDoIt( const G4Track& track,
                                                        const G4Step&  stepData )
{
    G4double stepLength  = track.GetStepLength();


    LOG(info) << Format(&stepData, m_origin, "trans.ASDIP.beg" );


    G4double preVelocity = stepData.GetPreStepPoint()->GetVelocity();

    G4double expectDeltaTime = 0.0 ;
    if ( preVelocity > 0.0 )  { expectDeltaTime = stepLength/preVelocity; } 
    G4double expectLocalTime =  track.GetLocalTime() + expectDeltaTime ;

    float wavelength = CTrack::Wavelength(&track);
    m_mlib->dumpGroupvelMaterial("trans.ASDIP.beg", wavelength, preVelocity,expectDeltaTime,  m_ctx._step_id );

    G4VParticleChange* change = G4Transportation::AlongStepDoIt(track, stepData );


    G4ParticleChangeForTransport* pcft = dynamic_cast<G4ParticleChangeForTransport*>(change) ;
    G4double changeLocalTime = pcft->GetLocalTime();
    assert( changeLocalTime == expectLocalTime );


    /*

    G4double startTime = track.GetGlobalTime() ;
    G4double postVelocity = stepData.GetPostStepPoint()->GetVelocity();
    G4double changeGlobalTime = pcft->GetGlobalTime();

    if(std::abs(wavelength-430.f) < 0.1f)
    {
        std::string preVelocityMat = m_mlib->firstMaterialWithGroupvelAt430nm( preVelocity, 0.001f );
        std::string postVelocityMat = m_mlib->firstMaterialWithGroupvelAt430nm( postVelocity, 0.001f );

        LOG(info) 
                  << " wavelength "           << std::setw(5) << wavelength
                  << " preVelocity "          << std::setw(10) << preVelocity
                  << " preVelocityMat(used) " << std::setw(20) << preVelocityMat 
                  << " postVelocity "         << std::setw(10) << postVelocity
                  << " postVelocityMat "      << std::setw(20) << postVelocityMat
                  << " startTime "            << std::setw(10) << startTime
                  << " stepLength "           << std::setw(10) << stepLength
                  << " expectLocalTime "      << std::setw(10) << expectLocalTime
                  << " changeLocalTime "      << std::setw(10) << changeLocalTime
                  << " changeGlobalTime "     << std::setw(10) << changeGlobalTime
                  << " expectDeltaTime "      << std::setw(10) << expectDeltaTime
                  ;
    }

    */


    return change ; 
}

