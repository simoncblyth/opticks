#pragma once

#include "G4OpBoundaryProcess.hh"
class G4Step ; 
class G4StepPoint ; 

class State 
{
   public:
       State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat );
       virtual ~State();
   public:
       const G4Step* getStep() const ;  
       G4OpBoundaryProcessStatus getBoundaryStatus() const ;
       const G4StepPoint* getPreStepPoint() const ;
       const G4StepPoint* getPostStepPoint() const ;
       unsigned int getPreMaterial() const ;
       unsigned int getPostMaterial() const ;
   private:
       const G4Step*             m_step ;
       G4OpBoundaryProcessStatus m_boundary_status ;
       unsigned int              m_premat ;
       unsigned int              m_postmat ;
};



inline const G4Step* State::getStep() const 
{
    return m_step ; 
}
inline G4OpBoundaryProcessStatus State::getBoundaryStatus() const
{
    return m_boundary_status ; 
}
inline unsigned int State::getPreMaterial() const 
{
    return m_premat ; 
}
inline unsigned int State::getPostMaterial() const
{
    return m_postmat ; 
}




