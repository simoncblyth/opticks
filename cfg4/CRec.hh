#pragma once

#include <vector>
#include "G4ThreeVector.hh"

#include "CBoundaryProcess.hh" 
#include "CStage.hh"

class Opticks ; 
class CStp ; 
class CPoi ; 
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

        bool is_step_limited() const ; 
        void setOrigin(const G4ThreeVector& origin);
        void clear();

        void dump(const char* msg="CRec::dump");

        unsigned getNumStp();
        CStp* getStp(unsigned index);

        unsigned getNumPoi();
        CPoi* getPoi(unsigned index);



#ifdef USE_CUSTOM_BOUNDARY
        bool add(DsG4OpBoundaryProcessStatus boundary_status);
#else
        bool add(G4OpBoundaryProcessStatus boundary_status);
#endif


#ifdef USE_CUSTOM_BOUNDARY
        void addPoi(DsG4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point, bool first);
#else
        void addPoi(G4OpBoundaryProcessStatus boundary_status,   const G4StepPoint* point, bool first );
#endif


#ifdef USE_CUSTOM_BOUNDARY
        void add(DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action);
#else
        void add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action);
#endif
    private:
        CG4*                        m_g4 ; 
        CG4Ctx&                     m_ctx ; 
        Opticks*                    m_ok ;  
        bool                        m_step_limited ; 
    private:
        G4ThreeVector               m_origin ; 
        std::vector<CStp*>          m_stp ; 
        std::vector<CPoi*>          m_poi ; 

};

#include "CFG4_TAIL.hh"

