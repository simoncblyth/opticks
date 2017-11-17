
#pragma once

#include <string>

#include "G4ThreeVector.hh" 
class G4StepPoint ; 

#include "CStage.hh"
#include "CBoundaryProcess.hh"

#include "CFG4_API_EXPORT.hh"

class CFG4_API CPoi 
{
    public:
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus getBoundaryStatus() const ;
        CPoi(const G4StepPoint* point, unsigned flag, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
#else
        G4OpBoundaryProcessStatus getBoundaryStatus() const  ;
        CPoi(const G4StepPoint* point, unsigned flag,  G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
#endif
        const G4StepPoint* getPoint() const ; 
        unsigned           getFlag() const ; 
        CStage::CStage_t   getStage() const ;

    private: 
        const G4StepPoint*  m_point ; 
        unsigned            m_flag ; 


#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
        G4OpBoundaryProcessStatus   m_boundary_status ;
#endif
        CStage::CStage_t  m_stage ;
        int               m_action ; 
        G4ThreeVector     m_origin ;

};
