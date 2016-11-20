#include <climits>

#include "CRec.hh"
#include "CStp.hh"
#include "Format.hh"

#include "PLOG.hh"

CRec::CRec(Opticks* ok, CGeometry* geometry, bool dynamic)
   :
    m_ok(ok), 
    m_geometry(geometry),
    m_material_bridge(NULL),
    m_dynamic(dynamic),
    m_record_id(UINT_MAX)
{
}
 

void CRec::startPhoton(unsigned record_id, const G4ThreeVector& origin)
{
    LOG(debug) << "CRec::startPhoton" 
              << " record_id " << record_id
              << " " << Format(origin, "origin")
              ;

    m_record_id = record_id ; 
    m_origin = origin ; 

    clearStp();
}


void CRec::clearStp()
{
    LOG(debug) << "CRec::clearStp"
              << " clearing " << m_stp.size() << " stps "
              ;
 
    m_stp.clear();
}



#ifdef USE_CUSTOM_BOUNDARY
void CRec::add(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag,  CStage::CStage_t stage, int action)
#else
void CRec::add(const G4Step* step, int step_id,  G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action)
#endif
{
    m_stp.push_back(new CStp(step, step_id, boundary_status, premat, postmat, preflag, postflag, stage, action, m_origin ));
}



unsigned CRec::getNumStps()
{
    return m_stp.size();
}

CStp* CRec::getStp(unsigned index)
{
    return m_stp[index]; 
}



void CRec::dump(const char* msg)
{
    unsigned nstp = m_stp.size();
    LOG(info) << msg  
              << " record_id " << m_record_id
              << " nstp " << nstp 
              << " " << ( nstp > 0 ? m_stp[0]->origin() : "-" ) 
              ; 

    for( unsigned i=0 ; i < nstp ; i++)
        std::cout << "(" << std::setw(2) << i << ") " << m_stp[i]->description() << std::endl ;  

}



