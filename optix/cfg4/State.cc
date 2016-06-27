#include "CFG4_BODY.hh"


#include "State.hh"
#include "G4Step.hh"



const G4Step* State::getStep() const 
{
    return m_step ; 
}
G4OpBoundaryProcessStatus State::getBoundaryStatus() const
{
    return m_boundary_status ; 
}
unsigned int State::getPreMaterial() const 
{
    return m_premat ; 
}
unsigned int State::getPostMaterial() const
{
    return m_postmat ; 
}



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


