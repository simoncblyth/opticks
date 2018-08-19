#include "CFG4_BODY.hh"


#include "State.hh"
#include "G4Step.hh"



const G4Step* State::getStep() const 
{
    return m_step ; 
}

#ifdef USE_CUSTOM_BOUNDARY
Ds::DsG4OpBoundaryProcessStatus State::getBoundaryStatus() const
#else
G4OpBoundaryProcessStatus State::getBoundaryStatus() const
#endif
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


CStage::CStage_t State::getStage() const 
{
    return m_stage ; 
}

unsigned State::getAction() const 
{
    return m_action ; 
}



#ifdef USE_CUSTOM_BOUNDARY
State::State(const G4Step* step, Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat, CStage::CStage_t stage, unsigned action) 
#else
State::State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat, CStage::CStage_t stage, unsigned action) 
#endif
   :
   m_step(NULL),
   m_boundary_status(boundary_status),
   m_premat(premat),
   m_postmat(postmat),
   m_stage(stage),
   m_action(action)
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


