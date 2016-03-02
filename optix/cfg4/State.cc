#include "State.hh"
#include "G4Step.hh"

State::State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat) 
   :
   m_step(NULL),
   m_boundary_status(boundary_status),
   m_premat(premat),
   m_postmat(postmat)
{
   m_step = new G4Step(*step) ;
}

State::~State()
{
   delete m_step ; 
}


const G4StepPoint* State::getPreStepPoint() const 
{
   return m_step->GetPreStepPoint();
}
const G4StepPoint* State::getPostStepPoint() const
{
   return m_step->GetPostStepPoint();
}


