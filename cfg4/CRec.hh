#pragma once

#include <vector>
#include "G4ThreeVector.hh"

#include "CBoundaryProcess.hh" 
#include "CStage.hh"

class Opticks ; 
class CGeometry ; 
class CMaterialBridge ; 
class CStp ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRec 
{
    public:
        CRec(Opticks* ok, CGeometry* geometry, bool dynamic);
        void startPhoton(unsigned record_id, const G4ThreeVector& origin);
        void dump(const char* msg="CRec::dump");

#ifdef USE_CUSTOM_BOUNDARY
        void add(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action);
#else
        void add(const G4Step* step, int step_id,  G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action);
#endif
    private:
        Opticks*                    m_ok ;  
        CGeometry*                  m_geometry ; 
        CMaterialBridge*            m_material_bridge ; 
        bool                        m_dynamic ; 
    private:
        unsigned                    m_record_id ; 
        G4ThreeVector               m_origin ; 
        std::vector<CStp*>          m_stp ; 

};

#include "CFG4_TAIL.hh"

