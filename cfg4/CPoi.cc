#include <sstream>

#include "G4StepPoint.hh"
#include "CPoi.hh"
#include "CAction.hh"
#include "OpticksFlags.hh"
#include "OpStatus.hh"

#include "Format.hh"



#ifdef USE_CUSTOM_BOUNDARY
CPoi::CPoi(const G4StepPoint* point, unsigned flag, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin) 
#else
CPoi::CPoi(const G4StepPoint* point, unsigned flag,   G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin) 
#endif
   :
   m_point(new G4StepPoint(*point)), 
   m_flag(flag),
   m_boundary_status(boundary_status),
   m_stage(stage),
   m_action(0),
   m_origin(origin)
{
}


const G4StepPoint* CPoi::getPoint() const 
{
    return m_point ; 
}

unsigned CPoi::getFlag() const 
{
    return m_flag ;
}

#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CPoi::getBoundaryStatus() const
#else
G4OpBoundaryProcessStatus   CPoi::getBoundaryStatus() const
#endif
{
   return m_boundary_status ;  
}
CStage::CStage_t CPoi::getStage() const
{
   return m_stage ; 
}




