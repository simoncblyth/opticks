#include <climits>


#include "OpticksPhoton.h"
#include "OpStatus.hh"


#include "CG4.hh"
#include "CG4Ctx.hh"

#include "CStp.hh"
#include "CPoi.hh"
#include "CRec.hh"
#include "CMaterialBridge.hh"

#include "Format.hh"

#include "PLOG.hh"

CRec::CRec(CG4* g4)
   :
    m_g4(g4),
    m_ctx(g4->getCtx()),
    m_ok(g4->getOpticks()),
    m_step_limited(false),
    m_material_bridge(NULL),
    m_prior_boundary_status(Undefined),
    m_boundary_status(Undefined)
{
}

void CRec::setMaterialBridge(CMaterialBridge* material_bridge) 
{
    m_material_bridge = material_bridge ; 
}



#ifdef USE_CUSTOM_BOUNDARY
void CRec::setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status)
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
    LOG(info) << msg  
              << " record_id " << m_ctx._record_id
              << " " << Format(m_origin, "origin")
              << " nstp " << nstp 
              << " " << ( nstp > 0 ? m_stp[0]->origin() : "-" ) 
              ; 


    for( unsigned i=0 ; i < nstp ; i++)
        std::cout << "(" << std::setw(2) << i << ") " << m_stp[i]->description() << std::endl ;  

}




void CRec::clear()
{
    if(m_ctx._dbgrec) 
    LOG(info) << "[--dbgrec] CRec::clear"
              << " stp " << m_stp.size() 
              << " poi " << m_poi.size() 
              ;
 
    m_stp.clear();
    m_poi.clear();
    m_step_limited = false ; 
}






#ifdef USE_CUSTOM_BOUNDARY
bool CRec::add(DsG4OpBoundaryProcessStatus boundary_status )
#else
bool CRec::add(G4OpBoundaryProcessStatus boundary_status )
#endif
{
    setBoundaryStatus(boundary_status);

    unsigned num_steps = m_stp.size() ;
    unsigned step_limit = m_ctx.step_limit() ;
    bool done = num_steps >= step_limit ;
  
/* 
    LOG(info) << "CRec::add"
              << " num_steps " << num_steps
              << " step_limit " << step_limit 
              << " done " << done 
              ;
*/

    if(done)
    {
        m_step_limited = true ; 
    } 
    else
    {
        m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, m_boundary_status, m_ctx._stage, m_origin));

        const G4Step* step = m_ctx._step ;
        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        unsigned preFlag = pointFlag( m_prior_boundary_status, pre );
        //assert( preFlag ); // tis zero for 1st when Undefined 

        unsigned postFlag = pointFlag( boundary_status, post );
        assert( postFlag );

        unsigned preMat = m_material_bridge->getPreMaterial(step) ; 
        unsigned postMat = m_material_bridge->getPostMaterial(step) ; 

        bool lastPre = OpStatus::IsTerminalFlag(preFlag);
        bool lastPost = OpStatus::IsTerminalFlag(postFlag);

        assert(!lastPre);

        bool preSkip = preFlag == NAN_ABORT || preFlag == 0  ; // StepTooSmall from BR
        bool matSwap = postFlag == NAN_ABORT ; // StepTooSmall coming up next, which will be preSkip 

        unsigned u_preMat  = matSwap ? postMat : preMat ;
        unsigned u_postMat = ( matSwap || postMat == 0 )  ? preMat  : postMat ;

        
       // canned style  :  pre+post,post,post,...   (with canned style can look into future when need arises)
       // live   style  :  pre,pre,pre,pre+post     (with live style cannot look into future, so need to operate with pre to allow peeking at post)

        if(!preSkip)    
        {
            m_poi.push_back(new CPoi(pre, preFlag, u_preMat, m_prior_boundary_status, m_ctx._stage, m_origin));
        }

        if(lastPost)
        {
            m_poi.push_back(new CPoi(post, postFlag, u_postMat, m_boundary_status, m_ctx._stage, m_origin));
        }

    }
   
    return done  ; 
}



#ifdef USE_CUSTOM_BOUNDARY
unsigned CRec::pointFlag(DsG4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point)
#else
unsigned CRec::pointFlag( G4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point)
#endif
{
    CStage::CStage_t stage = m_ctx._stage == CStage::REJOIN ? CStage::RECOLL : m_ctx._stage  ; // avoid duping the RE 

    unsigned flag = m_ctx._stage == CStage::START ? m_ctx._gen : OpStatus::OpPointFlag(point, boundary_status, stage );

    //assert( flag );  

    return flag ; 
}









#ifdef USE_CUSTOM_BOUNDARY
void CRec::add(DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#else
void CRec::add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#endif
{
    m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, boundary_status, premat, postmat, preflag, postflag, m_ctx._stage, action, m_origin ));
}




