#include <climits>

#include "CG4.hh"
#include "CG4Ctx.hh"
#include "CRec.hh"
#include "CStp.hh"
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

void CRec::clearStp()
{
    if(m_ctx._dbgrec) 
    LOG(info) << "[--dbgrec] CRec::clearStp"
              << " clearing " << m_stp.size() << " stps "
              ;
 
    m_stp.clear();
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
    }
    
    return done  ; 
}


#ifdef USE_CUSTOM_BOUNDARY
void CRec::add(DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#else
void CRec::add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action)
#endif
{
    m_stp.push_back(new CStp(m_ctx._step, m_ctx._step_id, boundary_status, premat, postmat, preflag, postflag, m_ctx._stage, action, m_origin ));
}

unsigned CRec::getNumStps()
{
    return m_stp.size();
}

CStp* CRec::getStp(unsigned index)
{
    return index < m_stp.size() ? m_stp[index] : NULL ; 
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

