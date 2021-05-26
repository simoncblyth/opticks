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
#include <climits>
#include <iomanip>

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksPhoton.h"
#include "OpStatus.hh"

#include "CG4Ctx.hh"

#include "CStp.hh"
#include "CPoi.hh"
#include "CAction.hh"
#include "CRec.hh"
#include "CRecState.hh"
#include "CMaterialBridge.hh"

#include "Format.hh"

#include "PLOG.hh"

const plog::Severity CRec::LEVEL = PLOG::EnvLevel("CRec", "DEBUG") ; 


CRec::CRec(CG4Ctx& ctx , CRecState& state)
    :
    m_state(state),
    m_ctx(ctx),
    m_ok(ctx.getOpticks()),
    m_recpoi(m_ok->isRecPoi()),             // --recpoi
    m_recpoialign(m_ok->isRecPoiAlign()),   // --recpoialign
    m_step_limited(false),
    m_point_done(false),
    m_material_bridge(NULL),
#ifdef USE_CUSTOM_BOUNDARY
    m_prior_boundary_status(Ds::Undefined),
    m_boundary_status(Ds::Undefined)
#else
    m_prior_boundary_status(Undefined),
    m_boundary_status(Undefined)
#endif
{
}

std::string CRec::desc() const 
{
    std::stringstream ss ; 
    ss << "CRec" 
       << " (" <<  ( m_recpoi ? "recpoi" : "recstp"  ) << ") "
       << " numStp " 
       << std::setw(2) 
       << getNumStp()
       << " step_limit " 
       << std::setw(2) 
       << m_ctx.step_limit()
       << " " << ( m_step_limited ? "STEP_LIMTED" : "-" )
       << " numPoi " 
       << std::setw(2) 
       << getNumPoi()
       << " point_limit " 
       << std::setw(2) 
       << m_ctx.point_limit()
       << ( m_point_done ? "POINT_DONE" : "-" )
       ;

    return ss.str();
}


void CRec::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
{
    assert(evt);
    if(m_recpoi)
    {
        evt->appendNote( "recpoi");
        evt->appendNote( m_recpoialign ? "recpoialign" : "not-aligned" );
    }
    else
    {
        evt->appendNote( "recstp");    //  b.metadata.Note from ab.py 
    }

    std::string note = evt->getNote();

    LOG(LEVEL) << "note " << note ; 

}




void CRec::setMaterialBridge(const CMaterialBridge* material_bridge) 
{
    m_material_bridge = material_bridge ; 
}



#ifdef USE_CUSTOM_BOUNDARY
void CRec::setBoundaryStatus(Ds::DsG4OpBoundaryProcessStatus boundary_status)
#else
void CRec::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status)
#endif
{
    m_prior_boundary_status = m_boundary_status ; 
    m_boundary_status = boundary_status ; 
}

 
bool CRec::is_step_limited() const
{
    return m_step_limited ;
}

/*
bool CRec::is_point_limited() const
{
    return m_point_limited ;
}


bool CRec::is_limited() const
{
    return m_recpoi ? m_point_limited : m_step_limited  ;
}

*/


void CRec::setOrigin(const G4ThreeVector& origin)
{
    m_origin = origin ; 
}

unsigned CRec::getNumStp() const 
{
    return m_stp.size();
}
CStp* CRec::getStp(unsigned index) const 
{
    return index < m_stp.size() ? m_stp[index] : NULL ; 
}

unsigned CRec::getNumPoi() const 
{
    return m_poi.size();
}
CPoi* CRec::getPoi(unsigned index) const 
{
    return index < m_poi.size() ? m_poi[index] : NULL ; 
}
CPoi* CRec::getPoiLast() const
{
    return m_poi.size() > 0 ? getPoi( m_poi.size() - 1 ) : NULL ; 
}



void CRec::dump(const char* msg)
{
    unsigned nstp = m_stp.size();
    unsigned npoi = m_poi.size();

    LOG(info) << msg  
              << " record_id " << m_ctx._record_id
              << " " << Format(m_origin, "origin")
              << " " << ( nstp > 0 ? m_stp[0]->origin() : "-" ) 
              ;

    LOG(info) << " nstp " << nstp ;
    for( unsigned i=0 ; i < nstp ; i++)
        std::cout << "(" << std::setw(2) << i << ") " << m_stp[i]->description() << std::endl ;  

    LOG(info) << " npoi " << npoi ;
    for( unsigned i=0 ; i < npoi ; i++)
        std::cout << "(" << std::setw(2) << i << ") " << m_poi[i]->description() << std::endl ;  

}



/**
CRec::clear
------------

NB explicit delete, skipping these and not having proper dtors for CStp and CPoi 
caused terrible leaking see notes/issues/plugging-cfg4-leaks.rst

**/

void CRec::clear()
{
    if(m_ctx._dbgrec) 
    LOG(info) << "[--dbgrec] CRec::clear"
              << " stp " << m_stp.size() 
              << " poi " << m_poi.size() 
              ;


    for(unsigned i=0 ; i < m_stp.size() ; i++)
    {
        CStp* stp = m_stp[i] ; 
        delete stp ;  
    }   
    m_stp.clear();

    for(unsigned i=0 ; i < m_poi.size() ; i++)
    {
        CPoi* poi = m_poi[i] ; 
        delete poi ;  
    }   
    m_poi.clear();

    m_step_limited = false ; 
    m_point_done = false ; 
}


/**
CRec::add
-----------

CRec::add is step-by-step invoked from CRecorder::Record
returning true kills the track, as needed for truncation of big bouncers
This copies the current step from context.

m_step_limited
    becomes true when collected steps reaches CG4Ctx::step_limit() which is 
    about twice the size you might expect because of StepTooSmall turnarounds : 
    this means that a photon will typically reach m_point_limited long before 
    it reaches m_step_limited 
    
m_point_limited
    becomes true when collected points reaches CG4Ctx::point_limit() which 
    is what you would expect : the larger of bouncemax and points to record

m_recpoialign
    aligns recpoi truncation according to step limit so both recpoi and recstp
    kill the track at same juncture.  This means the kill will happen later
    that with just "--recpoi"  

    This spins G4 wheels with the more efficient recpoi 
    in order to keep random sequence aligned with the less efficient !recpoi(aka recstp)
    see notes/issues/cfg4-recpoi-recstp-insidious-difference.rst

**/

#ifdef USE_CUSTOM_BOUNDARY
bool CRec::add(Ds::DsG4OpBoundaryProcessStatus boundary_status )
#else
bool CRec::add(G4OpBoundaryProcessStatus boundary_status )
#endif
{
    //m_ok->accumulateStart(m_add_acc); 

    setBoundaryStatus(boundary_status);

    m_step_limited = m_stp.size() >= m_ctx.step_limit() ;    
    //m_point_limited = m_poi.size() >= m_ctx.point_limit() ;

    CStp* stp = new CStp(m_ctx._step, m_ctx._step_id, m_boundary_status, m_ctx._stage, m_origin) ;
    m_stp.push_back(stp);

    // collect points from each step until reach termination or point limit 
    if(m_recpoi && !m_point_done )  
    {
        m_point_done = addPoi(stp) ;  // <-- point_done happens from a lastPost with a terminal flag OR fill point limit 
    }

    // hmm the point_limited is based on the size before 
    // so 11-pointers will cause a problem as they will add 2 

    bool done = ( m_recpoialign || !m_recpoi ) ? m_step_limited : m_point_done  ;

    //m_ok->accumulateStop(m_add_acc); 
    return done ;  // (*lldb*) add
}


/**
CRec::addPoi
--------------

Invoked by CRec::add when using the alternative m_recpoi mode 


There are two styles of pre/post recording, addPoi is using live style

canned 
   pre+post,post,post,...   (with canned style can look into future when need arises)
live   
   pre,pre,pre,pre+post     (with live style cannot look into future, so need to operate with pre to allow peeking at post)


* some pre are skipped as they are G4 StepTooSmall technicalities that happen at reflections
* at lastPost pre+post are appended to m_poi

**/

bool CRec::addPoi(CStp* stp )
{
    m_state._step_action = 0 ; 

    switch(m_ctx._stage)
    {
        case CStage::START:  m_state._step_action |= CAction::STEP_START    ; break ; 
        case CStage::REJOIN: m_state._step_action |= CAction::STEP_REJOIN   ; break ; 
        case CStage::RECOLL: m_state._step_action |= CAction::STEP_RECOLL   ; break ;
        case CStage::COLLECT:                                               ; break ; 
        case CStage::UNKNOWN:assert(0)                                      ; break ; 
    } 

    const G4Step* step = m_ctx._step ;
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 


    CStage::CStage_t stage = m_ctx._stage == CStage::REJOIN ? CStage::RECOLL : m_ctx._stage  ; // avoid duping the RE 

    unsigned preFlag = stage == CStage::START ? 
                                                 m_ctx._gen
                                              : 
                                                 OpStatus::OpPointFlag(pre, m_prior_boundary_status, stage )
                                              ;

    unsigned postFlag = OpStatus::OpPointFlag(post, m_boundary_status, stage ) ;  // only stage REJOIN yields BULK_REEMIT

    assert( preFlag ); 
    assert( postFlag );

    unsigned preMat = m_material_bridge->getPreMaterial(step) ; 
    unsigned postMat = m_material_bridge->getPostMaterial(step) ; 

    bool lastPre = OpStatus::IsTerminalFlag(preFlag);  assert(!lastPre);
    bool lastPost = OpStatus::IsTerminalFlag(postFlag);

    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

#ifdef USE_CUSTOM_BOUNDARY
    bool preSkip = m_prior_boundary_status == Ds::StepTooSmall && m_ctx._stage != CStage::REJOIN  && m_ctx._stage != CStage::START  ;  
#else
    bool preSkip = m_prior_boundary_status == StepTooSmall && m_ctx._stage != CStage::REJOIN  && m_ctx._stage != CStage::START  ;  
#endif
    // CStage::START is never preSkip as want to record the m_ctx._gen eg "TO" into zeroth flag 

    bool matSwap = postFlag == NAN_ABORT ; // StepTooSmall coming up next, which will be preSkip 

    if(lastPost)      m_state._step_action |= CAction::LAST_POST ; 
    if(surfaceAbsorb) m_state._step_action |= CAction::SURF_ABS ;  
    if(preSkip)       m_state._step_action |= CAction::PRE_SKIP ; 
    if(matSwap)       m_state._step_action |= CAction::MAT_SWAP ; 

    unsigned u_preMat  = matSwap ? postMat : preMat ;
    unsigned u_postMat = ( matSwap || postMat == 0 )  ? preMat  : postMat ;


    bool limited = false ; 

    if(!preSkip)    
    {
        limited = addPoi_(new CPoi(pre, preFlag, u_preMat, m_prior_boundary_status, m_ctx._stage, m_origin));
    }

    if(lastPost && !limited)
    {
        limited = addPoi_(new CPoi(post, postFlag, u_postMat, m_boundary_status, m_ctx._stage, m_origin));
    }

    if(stp) // debug notes for dumping
    {
        stp->setMat(  u_preMat, u_postMat );
        stp->setFlag( preFlag,  postFlag );
        stp->setAction( m_state._step_action );
    }

    bool done = lastPost || limited ; 
    return done  ;   
}


/**
CRec::addPoi_
-----------------

Hmm : limiting like this is not the real solution, because it 
conflates bounce_max with record_max and prevents matching 
final photon unless bounce_max is less than record_max 


**/


bool CRec::addPoi_(CPoi* poi)
{
    bool limited = m_poi.size() >= m_ctx.point_limit() ;
    if( !limited )
    {
        m_poi.push_back(poi); 
    }
    return limited  ; 
}




#ifdef USE_CUSTOM_BOUNDARY
void CRec::add(Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#else
void CRec::add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#endif
{
    m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, boundary_status, premat, postmat, preflag, postflag, m_ctx._stage, action, m_origin ));
}




