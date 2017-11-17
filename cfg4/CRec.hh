#pragma once

#include <vector>
#include "G4ThreeVector.hh"

#include "CBoundaryProcess.hh" 
#include "CStage.hh"

class Opticks ; 
class CStp ; 
class CPoi ; 
class CG4 ; 
class CMaterialBridge ; 

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
        void setMaterialBridge(CMaterialBridge* material_bridge) ;

        void dump(const char* msg="CRec::dump");

        unsigned getNumStp() const ;
        CStp* getStp(unsigned index) const ;

        unsigned getNumPoi() const ;
        CPoi* getPoi(unsigned index) const ;
        CPoi* getPoiLast() const ;

   public:

#ifdef USE_CUSTOM_BOUNDARY
        bool add(DsG4OpBoundaryProcessStatus boundary_status);
#else
        bool add(G4OpBoundaryProcessStatus boundary_status);
#endif


   private:
#ifdef USE_CUSTOM_BOUNDARY
        void setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status);
#else
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status);
#endif

#ifdef USE_CUSTOM_BOUNDARY
        unsigned pointFlag(DsG4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point);
#else
        unsigned pointFlag( G4OpBoundaryProcessStatus boundary_status, const G4StepPoint* point);
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
        CMaterialBridge*            m_material_bridge ; 
    private:
        G4ThreeVector               m_origin ; 
        std::vector<CStp*>          m_stp ; 
        std::vector<CPoi*>          m_poi ; 

   private:
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus m_prior_boundary_status ; 
        DsG4OpBoundaryProcessStatus m_boundary_status ; 
#else
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 
        G4OpBoundaryProcessStatus m_boundary_status ; 
#endif




};

#include "CFG4_TAIL.hh"

