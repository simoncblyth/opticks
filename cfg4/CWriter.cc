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
#include "CGenstep.hh"
#include "CCtx.hh"
#include "CPhoton.hh"
#include "Opticks.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "CWriter.hh"
#include "PLOG.hh"

const plog::Severity CWriter::LEVEL = PLOG::EnvLevel("CWriter", "DEBUG") ; 

CWriter::CWriter(CCtx& ctx, CPhoton& photon)
    :
    m_photon(photon),
    m_ctx(ctx),
    m_ok(m_ctx.getOpticks()),
    m_enabled(true),
    m_evt(NULL),

    m_records_buffer(NULL),
    m_photons_buffer(NULL),
    m_history_buffer(NULL)
{
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

    m_evt->setDynamic(1) ;  

    LOG(LEVEL) 
        << " _record_max " << m_ctx._record_max
        << " _bounce_max  " << m_ctx._bounce_max 
        << " _steps_per_photon " << m_ctx._steps_per_photon 
        << " num_g4event " << m_evt->getNumG4Event() 
        ;

    m_history_buffer = m_evt->getSequenceData();  // ph : seqhis/seqmat
    m_photons_buffer = m_evt->getPhotonData();    // ox : final photon
    m_records_buffer = m_evt->getRecordData();    // rx :  step records

    LOG(LEVEL) << desc() ; 
}

std::string CWriter::desc(const char* msg) const 
{
    assert( m_history_buffer ); 
    std::stringstream ss ; 
    if(msg) ss << msg << " " ; 
    ss << " m_history_buffer " << m_history_buffer->getShapeString() ; 
    ss << " m_photons_buffer " << m_photons_buffer->getShapeString() ; 
    ss << " m_records_buffer " << m_records_buffer->getShapeString() ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
CWriter::expand
----------------

Invoked by CWriter::BeginOfGenstep


**/
unsigned CWriter::expand(unsigned gs_photons)
{
    if(!m_history_buffer) 
    {
        LOG(fatal) << " Cannot expand as CWriter::initEvent has not been called, check CManager logging " ;
        return 0 ;  
    } 
    assert( m_history_buffer );  
    unsigned ni, ni1, ni2 ; 
    ni = m_history_buffer->expand(gs_photons); 
    ni1 = m_photons_buffer->expand(gs_photons); 
    ni2 = m_records_buffer->expand(gs_photons); 
    assert( ni1 == ni && ni2 == ni ); 
    return ni ; 
}


/**
CWriter::BeginOfGenstep
-------------------------

Invoked from CRecorder::BeginOfGenstep, expands the buffers to accomodate the photons of this genstep.

**/

void CWriter::BeginOfGenstep()
{
    unsigned genstep_num_photons =  m_ctx._genstep_num_photons ; 
    m_ni = expand(genstep_num_photons);  

    LOG(LEVEL)
        << " m_ctx._gentype [" <<  m_ctx._gentype << "]" 
        << " m_ctx._genstep_index " << m_ctx._genstep_index
        << " m_ctx._genstep_num_photons " << m_ctx._genstep_num_photons
        << " m_ni " << m_ni 
        ;


}


/**
CWriter::writeStepPoint
------------------------
   
Invoked by CRecorder::postTrackWriteSteps/CRecorder::WriteStepPoint

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
    unsigned record_id = m_ctx._record_id ;  

    LOG(LEVEL)  
        << " m_ctx._photon_id " << m_ctx._photon_id 
        << " m_ctx._record_id " << m_ctx._record_id 
        << " m_ni " << m_ni 
        ;
  
    assert( record_id < m_ni ); 
    assert( m_records_buffer ); 


    m_photon.add(flag, material);  // sets seqhis/seqmat nibbles in current constrained slot  

    bool hard_truncate = m_photon.is_hard_truncate();    

    bool done = false ;   

    if(hard_truncate) 
    {
        done = true ; 
    }
    else
    {
        if(m_enabled) writeStepPoint_(point, m_photon, record_id  );

        m_photon.increment_slot() ; 

        done = m_photon.is_done() ;  // caution truncation/is_done may change after increment

        if( (done || last) && m_enabled )
        {
            writePhoton_(point, record_id );
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
A: the arrays are extended at BeginOfGenstep allowing 
   random access writes into the dynamically growing arrays 

The current implementation was adapted to remove the need
to track secondaries first.  Any order of tracks should now work 
just fine, assuming that secondaries always come after their parents.
Which should always the case (?)

Reemission RE-joining of tracks are handled by resuming the 
writing onto ncestor photon records lined up using the record_id.

**/

void CWriter::writeStepPoint_(const G4StepPoint* point, const CPhoton& photon, unsigned record_id  )
{
    unsigned slot = photon._slot_constrained ;
    unsigned flag = photon._flag ; 
    unsigned material = photon._mat ; 

    //assert(  record_id < m_ctx._record_max );

    if(m_ctx._dbgrec)
    {
        LOG(info) << "[--dbgrec]"
                  << " record_id " << record_id
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

    m_records_buffer->setQuad(record_id, slot, 0, posx, posy, posz, time_ );
    m_records_buffer->setQuad(record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

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

void CWriter::writePhoton_(const G4StepPoint* point, unsigned record_id  )
{
    assert( m_photons_buffer ); 
    writeHistory_(record_id);  

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ;  

    // emulating the Opticks GPU written photons 
    m_photons_buffer->setQuad(record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_photons_buffer->setQuad(record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    m_photons_buffer->setQuad(record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
    m_photons_buffer->setUInt(record_id, 3, 0, 0, m_photon._slot_constrained );
    m_photons_buffer->setUInt(record_id, 3, 0, 1, 0u );
    m_photons_buffer->setUInt(record_id, 3, 0, 2, m_photon._c4.u );
    m_photons_buffer->setUInt(record_id, 3, 0, 3, m_photon._mskhis );
}

/**
CWriter::writeHistory_
-----------------------

Emulating GPU seqhis/seqmat writing 

NB although pointers into the buffer are used here, crucially the pointers are 
not held and they are accessed fresh everytime as they will move around  
as data gets reallocated with array expansion.

**/

void CWriter::writeHistory_(unsigned record_id)
{
    assert( m_history_buffer ); 
    unsigned long long* history = m_history_buffer->getValues() + 2*record_id ; 
    *(history+0) = m_photon._seqhis ; 
    *(history+1) = m_photon._seqmat ; 
}

