#pragma once

// Holds copy of a step together with id 

class G4Step ; 

#include "CFG4_API_EXPORT.hh"
class CFG4_API CStep {
   public:
       CStep(const G4Step* step, unsigned int step_id);
       virtual ~CStep();
       const G4Step* getStep() const ;  
       unsigned int  getStepId() const ; 
   private:
       const G4Step* m_step ; 
       unsigned int  m_step_id ; 
};


