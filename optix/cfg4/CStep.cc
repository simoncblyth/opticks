#include "CStep.hh"

#include "State.hh"
#include "G4Step.hh"

CStep::CStep(const G4Step* step, unsigned int step_id) 
   :
   m_step(NULL),
   m_step_id(step_id)
{
   m_step = new G4Step(*step) ;
}

CStep::~CStep()
{
   delete m_step ; 
}


