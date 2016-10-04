#pragma once


#include "CFG4_PUSH.hh"
// huh: G4OpBoundaryProcess pulls in CLHEP/Random headers that raise warnings
#include "CBoundaryProcess.hh"
#include "CFG4_POP.hh"

class G4Step ; 
class G4StepPoint ; 


#include "CFG4_API_EXPORT.hh"
class CFG4_API State 
{
   public:
#ifdef USE_CUSTOM_BOUNDARY
       State(const G4Step* step, DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat );
       DsG4OpBoundaryProcessStatus getBoundaryStatus() const ;
#else
       State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat );
       G4OpBoundaryProcessStatus getBoundaryStatus() const ;
#endif
       virtual ~State();
   public:
       const G4Step* getStep() const ;  
       const G4StepPoint* getPreStepPoint() const ;
       const G4StepPoint* getPostStepPoint() const ;
       unsigned int getPreMaterial() const ;
       unsigned int getPostMaterial() const ;
   private:
       const G4Step*             m_step ;
#ifdef USE_CUSTOM_BOUNDARY
       DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
       G4OpBoundaryProcessStatus m_boundary_status ;
#endif
       unsigned int              m_premat ;
       unsigned int              m_postmat ;
};



