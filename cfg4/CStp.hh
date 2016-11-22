#pragma once

#include <string>

#include "G4ThreeVector.hh" 
class G4Step ; 

#include "CStage.hh"
#include "CBoundaryProcess.hh"

#include "CFG4_API_EXPORT.hh"

class CFG4_API CStp 
{
   public:
#ifdef USE_CUSTOM_BOUNDARY
       DsG4OpBoundaryProcessStatus getBoundaryStatus() ;
       CStp(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
       CStp(const G4Step* step, int step_id, DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action, const G4ThreeVector& origin);
#else
       G4OpBoundaryProcessStatus   getBoundaryStatus() ;
       CStp(const G4Step* step, int step_id,   G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
       CStp(const G4Step* step, int step_id,   G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action, const G4ThreeVector& origin); 
#endif
        std::string description();
        std::string origin();

    public:
         const G4Step*    getStep();
         int              getStepId(); 
         CStage::CStage_t getStage();

         void   setMat(  unsigned premat, unsigned postmat);
         void   setFlag( unsigned preflag, unsigned postflag );
         void   setAction(int action);

   private:
         const G4Step*               m_step ; 
         int                         m_step_id ; 
#ifdef USE_CUSTOM_BOUNDARY
         DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
         G4OpBoundaryProcessStatus   m_boundary_status ;
#endif
         unsigned          m_premat ; 
         unsigned          m_postmat ; 
         unsigned          m_preflag ; 
         unsigned          m_postflag ; 

         CStage::CStage_t  m_stage ;
         int               m_action ; 
         G4ThreeVector     m_origin ;

};

