#include <climits>


#include "OpStatus.hh"


#include "CG4.hh"
#include "CG4Ctx.hh"

#include "CStp.hh"
#include "CPoi.hh"
#include "CRec.hh"

#include "Format.hh"

#include "PLOG.hh"

CRec::CRec(CG4* g4)
   :
    m_g4(g4),
    m_ctx(g4->getCtx()),
    m_ok(g4->getOpticks()),
    m_step_limited(false)
{
}
 
bool CRec::is_step_limited() const
{
    return m_step_limited ;
}

void CRec::setOrigin(const G4ThreeVector& origin)
{
    m_origin = origin ; 
}

unsigned CRec::getNumStp()
{
    return m_stp.size();
}
CStp* CRec::getStp(unsigned index)
{
    return index < m_stp.size() ? m_stp[index] : NULL ; 
}

unsigned CRec::getNumPoi()
{
    return m_poi.size();
}
CPoi* CRec::getPoi(unsigned index)
{
    return index < m_poi.size() ? m_poi[index] : NULL ; 
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
        m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, boundary_status, m_ctx._stage, m_origin));

        // experimental point based recording ...
        const G4Step* step = m_ctx._step ;
        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        if( m_poi.size() == 0 )
        {
            addPoi( Undefined      , pre,  true );
            addPoi( boundary_status, post, false );
        }
        else
        {
            addPoi( boundary_status, post, false );
        }
    }
   
    return done  ; 
}



#ifdef USE_CUSTOM_BOUNDARY
void CRec::addPoi(DsG4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point, bool first)
#else
void CRec::addPoi(G4OpBoundaryProcessStatus boundary_status,   const G4StepPoint* point, bool first )
#endif
{
    unsigned flag = 0 ; 
    if( first ) 
    {
        flag = m_ctx._gen ; 
    }
    else
    {
        CStage::CStage_t stage = m_ctx._stage == CStage::REJOIN ? CStage::RECOLL : m_ctx._stage  ; // avoid duping the RE 
        flag = OpPointFlag(point, boundary_status, stage);
    }


    bool last = IsTerminalFlag(flag);
    bool skip = boundary_status == StepTooSmall && !last  ;  

    if(!skip)
    {
        m_poi.push_back(new CPoi(point, flag, boundary_status, m_ctx._stage, m_origin));
    }
}




#ifdef USE_CUSTOM_BOUNDARY
void CRec::add(DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#else
void CRec::add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#endif
{
    m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, boundary_status, premat, postmat, preflag, postflag, m_ctx._stage, action, m_origin ));
}




