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

#include <cassert>
#include <sstream>

#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"


#include "BConverter.hh"
#include "SBit.hh"

#include "CRecorder.h"
#include "CG4Ctx.hh"
#include "CPhoton.hh"
#include "Opticks.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "CWriter.hh"
#include "PLOG.hh"

const plog::Severity CWriter::LEVEL = PLOG::EnvLevel("CWriter", "DEBUG") ; 

CWriter::CWriter(CG4Ctx& ctx, CPhoton& photon, bool onestep)
    :
    m_photon(photon),
    m_onestep(onestep),
    m_ctx(ctx),
    m_ok(m_ctx.getOpticks()),
    m_enabled(true),
    m_evt(NULL),

    m_records_buffer(NULL),
    m_photons_buffer(NULL),
    m_history_buffer(NULL),

    m_onestep_records(NULL),
    m_onestep_photons(NULL),
    m_onestep_history(NULL),

    m_target_records(NULL),
    m_target_photons(NULL),
    m_target_history(NULL)
{
    LOG(LEVEL) << " " << ( m_onestep ? "ONESTEP" : "STATIC/ALLSTEP" ) ;
}

void CWriter::setEnabled(bool enabled)
{
    m_enabled = enabled ; 
}

/**
CWriter::initEvent
-------------------

Gets refs to the history, photons and records buffers from the event.
When dynamic the records target is single item dynamic_records otherwise
goes direct to the records_buffer.

**/

void CWriter::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());

    m_evt->setDynamic( m_onestep ? 1 : 0 ) ;  

    LOG(LEVEL) 
        << ( m_onestep ? "ONESTEP(CPU style)" : "STATIC(GPU style)" )
        << " _record_max " << m_ctx._record_max
        << " _bounce_max  " << m_ctx._bounce_max 
        << " _steps_per_photon " << m_ctx._steps_per_photon 
        << " num_g4event " << m_evt->getNumG4Event() 
        ;

    m_history_buffer = m_evt->getSequenceData();
    m_photons_buffer = m_evt->getPhotonData();
    m_records_buffer = m_evt->getRecordData();

    LOG(LEVEL) << desc() ; 

    assert( m_history_buffer && "CRecorder requires history buffer" );
    assert( m_photons_buffer && "CRecorder requires photons buffer" );
    assert( m_records_buffer && "CRecorder requires records buffer" );

    // these targets get set by CWriter::initGenstep in onestep running 
    m_target_history = nullptr ; 
    m_target_photons = nullptr ; 
    m_target_records = nullptr ;  

}


std::string CWriter::desc(const char* msg) const 
{
    std::stringstream ss ; 
    if(msg) ss << msg << " " ; 
    ss << ( m_onestep ? "ONESTEP(CPU style)" : "STATIC(GPU style)" ) ; 
    ss << " m_history_buffer " << m_history_buffer->getShapeString() ; 
    ss << " m_photons_buffer " << m_photons_buffer->getShapeString() ; 
    ss << " m_records_buffer " << m_records_buffer->getShapeString() ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
CWriter::initGenstep
----------------------

Invoked from CRecorder::initEvent.

1. creates small onestep buffers 
2. change target to point at them 

**/

void CWriter::initGenstep( char gentype, int num_onestep_photons )
{
    LOG(LEVEL) 
        << " gentype [" <<  gentype << "]" 
        << " num_onestep_photons " << num_onestep_photons 
        ; 

    assert( m_onestep ); 
    assert( m_ctx._gentype == gentype  ); 
    assert( m_ctx._genstep_num_photons == unsigned(num_onestep_photons)  ); 

    m_onestep_records = NPY<short>::make(num_onestep_photons, m_ctx._steps_per_photon, 2, 4) ;
    m_onestep_records->zero();

    m_onestep_photons = NPY<float>::make(num_onestep_photons, 4, 4) ;
    m_onestep_photons->zero();

    m_onestep_history = NPY<unsigned long long>::make(num_onestep_photons, 1, 2) ;
    m_onestep_history->zero();

    m_target_records = m_onestep_records ; 
    m_target_photons = m_onestep_photons ; 
    m_target_history = m_onestep_history ; 
}

/**
CWriter::writeGenstep
-----------------------

**/

void CWriter::writeGenstep( char gentype, int num_onestep_photons )
{
    LOG(LEVEL) << " gentype [" <<  gentype << "] num_onestep_photons " << num_onestep_photons ; 
    assert( m_onestep ); 

    // currently are going direct to the buffer, not into m_onestep_* first

    LOG(LEVEL) << desc("bef.add") ; 

    assert( m_records_buffer->getNumItems() == 0 );

    m_records_buffer->add(m_onestep_records);
    m_photons_buffer->add(m_onestep_photons);
    m_history_buffer->add(m_onestep_history);

    LOG(LEVEL) << desc("aft.add") ; 

    clearOnestep(); 
}

void CWriter::clearOnestep()
{
    m_onestep_records->reset(); 
    m_onestep_photons->reset(); 
    m_onestep_history->reset(); 

    delete m_onestep_records ; 
    delete m_onestep_photons ; 
    delete m_onestep_history ; 

    m_onestep_records = nullptr ; 
    m_onestep_photons = nullptr ; 
    m_onestep_history = nullptr ; 

    m_target_records = nullptr ; 
    m_target_photons = nullptr ; 
    m_target_history = nullptr ; 

}




/**
CWriter::writeStepPoint
------------------------
   
Invoked by CRecorder::WriteStepPoint

* writes into the target record, photon and history buffers

* writes point-by-point records
* when done writes final photons

* NB the "done" returned here **DOES NOT** kill tracks, 
  that happens on collection NOT on writing 

*hard_truncate* does happen for top slot without reemission rejoinders

* hmm: suspect its causing seqhis zeros ?


HMM : LOOKS LIKE SOME CONFLATION BETWEEN REAL bounce_max truncation 
and recording truncation 

* hard_truncate is only expected the 2nd time adding into top slot 

* *last* argument is only used in --recpoi mode where it prevents 
   truncated photons from never being "done" and giving seqhis zeros


**/     

bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last )
{
    m_photon.add(flag, material);  // sets seqhis/seqmat nibbles in current constrained slot  

    bool hard_truncate = m_photon.is_hard_truncate();    

    bool done = false ;   

    if(hard_truncate) 
    {
        done = true ; 
    }
    else
    {
        if(m_enabled) writeStepPoint_(point, m_photon );

        m_photon.increment_slot() ; 

        done = m_photon.is_done() ;  // caution truncation/is_done may change after increment

        if( (done || last) && m_enabled )
        {
            writePhoton(point);
        }
    }        

    if( flag == BULK_ABSORB )
    {
        assert( done == true ); 
    }

    return done ; 
}


/**
CWriter::writeStepPoint_
--------------------------

Writes compressed step records if Geant4 CG4 instrumented events 
in format to match that written on GPU by oxrap.

Q: how does this work with REJOIN and dynamic running ?
A: I think it doesnt work correctly yet ... need to make random 
   access to total number of photons of the genstep.

NB the use of m_ctx._record_id rather than using some stored copy 
of that is what is restricting this to strictly seeing all the steps for 
one photon sequentially. That including steps after one or more generations of reemission.
This is why Geant4 process must be configured to track secondaries first.

**/

void CWriter::writeStepPoint_(const G4StepPoint* point, const CPhoton& photon )
{
    // write compressed record quads into buffer at location for the m_record_id 

    unsigned target_record_id = m_onestep ? m_ctx._record_id : m_ctx._record_id ;   
    // hmm maybe separate id for onestep handling and all genstep handling 

    LOG(LEVEL)  << " target_record_id " << target_record_id ; 
    assert( m_target_records ); 

    unsigned slot = photon._slot_constrained ;
    unsigned flag = photon._flag ; 
    unsigned material = photon._mat ; 

    if(!m_onestep) assert( target_record_id < m_ctx._record_max );

    if(m_ctx._dbgrec)
    {
        LOG(info) << "[--dbgrec]"
                  << " target_record_id " << target_record_id
                  << " slot " << slot 
                  << " flag " << flag 
                  << " material " << material 
                  ;
    }  

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& pol = point->GetPolarization();
    //LOG(info) << " pos " <<  std::setw(30) << pos << " pol " <<  std::setw(30) << pol ; 

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;

    const glm::vec4& sd = m_evt->getSpaceDomain() ; 
    const glm::vec4& td = m_evt->getTimeDomain() ; 
    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 


    short posx = BConverter::shortnorm(pos.x()/mm, sd.x, sd.w ); 
    short posy = BConverter::shortnorm(pos.y()/mm, sd.y, sd.w ); 
    short posz = BConverter::shortnorm(pos.z()/mm, sd.z, sd.w ); 
    short time_ = BConverter::shortnorm(time/ns,   td.x, td.y );

    float wfrac = ((wavelength/nm) - wd.x)/wd.w ;   

    // see oxrap/cu/photon.h
    // tboolean-box : for first steppoint the pol is same as pos causing out-of-range
    // notes/issues/tboolean-resurrection.rst
    unsigned char polx = BConverter::my__float2uint_rn( (pol.x()+1.f)*127.f );
    unsigned char poly = BConverter::my__float2uint_rn( (pol.y()+1.f)*127.f );
    unsigned char polz = BConverter::my__float2uint_rn( (pol.z()+1.f)*127.f );
    unsigned char wavl = BConverter::my__float2uint_rn( wfrac*255.f );

    qquad qaux ; 
    qaux.uchar_.x = material ; 
    qaux.uchar_.y = 0 ; // TODO:m2 
    qaux.char_.z  = 0 ; // TODO:boundary (G4 equivalent ?)
    qaux.uchar_.w = SBit::ffs(flag) ;   // ? duplicates seqhis  

    hquad polw ; 
    polw.ushort_.x = polx | poly << 8 ; 
    polw.ushort_.y = polz | wavl << 8 ; 
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;

    //unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 

    m_target_records->setQuad(target_record_id, slot, 0, posx, posy, posz, time_ );
    m_target_records->setQuad(target_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    // dynamic mode : fills in slots into single photon dynamic_records structure 
    // static mode  : fills directly into a large fixed dimension records structure

    // looks like static mode will succeed to scrub the AB and replace with RE 
    // just by decrementing m_slot and running again
    // but dynamic mode will have an extra record
}


/**
CWriter::writePhoton
--------------------

Gets called at last step (eg absorption) or when truncated,
for reemission have to rely on downstream overwrites
via rerunning with a target_record_id to scrub old values.

In static case (where the number of photons is known ahead of time) 
directly populates the pre-sized photon buffer in dynamic case populates 
the single item m_dynamic_photons buffer first and then adds that to 
the m_photons_buffer

generate.cu

(x)  p.flags.i.x = prd.boundary ;   // last boundary
(y)  p.flags.u.y = s.identity.w ;   // sensorIndex  >0 only for cathode hits
(z)  p.flags.u.z = s.index.x ;      // material1 index  : redundant with boundary  
(w)  p.flags.u.w |= s.flag ;        // OR of step flags : redundant ? unless want to try to live without seqhis


Q: Does RE-joining of reemission work with dynamic running ?

A: Hmm, not sure : but suspect that is could be made to work by using 
   SetTrackSecondariesFirst::

       cerenkov->SetTrackSecondariesFirst(true);
       scint->SetTrackSecondariesFirst(true);    

   With reemission there is only one secondary, so deferring writePhoton 
   until get a new one that is not a REjoinder means that do not need to 
   have random access to the full photon/record buffers.

   That could work with multiple RE in the history, so long as the SetTrackSecondariesFirst 
   worked to simulate the generations one after the other.      

**/

void CWriter::writePhoton(const G4StepPoint* point )
{
    unsigned target_record_id = m_onestep ? m_ctx._record_id : m_ctx._record_id ; 
    LOG(LEVEL)  << " target_record_id " << target_record_id ; 
    assert( m_target_photons ); 
    writeHistory(target_record_id);  


    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ;  

    // emulating the Opticks GPU written photons 
    m_target_photons->setQuad(target_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_target_photons->setQuad(target_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    m_target_photons->setQuad(target_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
    m_target_photons->setUInt(target_record_id, 3, 0, 0, m_photon._slot_constrained );
    m_target_photons->setUInt(target_record_id, 3, 0, 1, 0u );
    m_target_photons->setUInt(target_record_id, 3, 0, 2, m_photon._c4.u );
    m_target_photons->setUInt(target_record_id, 3, 0, 3, m_photon._mskhis );
}

/**
CWriter::writeHistory
-----------------------

Emulating GPU seqhis/seqmat writing 

**/

void CWriter::writeHistory(unsigned target_record_id)
{
    assert( m_target_history ); 
    unsigned long long* history = m_target_history->getValues() + 2*target_record_id ;
    *(history+0) = m_photon._seqhis ; 
    *(history+1) = m_photon._seqmat ; 
}

