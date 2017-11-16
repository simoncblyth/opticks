#pragma once

#include <vector>
#include "G4ThreeVector.hh"

#include "CBoundaryProcess.hh" 
#include "CStage.hh"

class Opticks ; 
class CStp ; 
class CG4 ; 

struct CG4Ctx ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CRec
=====

Canonical m_crec instance is resident of CRecorder and is instanciated with it.

**/

class CFG4_API CRec 
{
    public:
        CRec(CG4* g4);

        // invoked by CRecorder::startPhoton, invokes clearStep
        void setOrigin(const G4ThreeVector& origin);
        // clears the added steps
        void clearStp();

        void dump(const char* msg="CRec::dump");
        unsigned getNumStps();
        CStp* getStp(unsigned index);


#ifdef USE_CUSTOM_BOUNDARY
        bool add(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage );
        void add(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action);
#else
        bool add(const G4Step* step, int step_id,  G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage );
        void add(const G4Step* step, int step_id,  G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action);
#endif
    private:
        CG4*                        m_g4 ; 
        CG4Ctx&                     m_ctx ; 
        Opticks*                    m_ok ;  
    private:
        G4ThreeVector               m_origin ; 
        std::vector<CStp*>          m_stp ; 

};

#include "CFG4_TAIL.hh"

