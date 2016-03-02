#pragma once

#include "G4OpBoundaryProcess.hh"
class G4Step ; 
class G4StepPoint ; 

class State 
{
   public:
       State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat );

       G4OpBoundaryProcessStatus getBoundaryStatus() const ;
       const G4StepPoint* getPreStepPoint() const ;
       const G4StepPoint* getPostStepPoint() const ;

       virtual ~State();

       const G4Step*             m_step ;
       G4OpBoundaryProcessStatus m_boundary_status ;
       unsigned int              m_premat ;
       unsigned int              m_postmat ;
};


inline G4OpBoundaryProcessStatus State::getBoundaryStatus() const
{
    return m_boundary_status ; 
}




