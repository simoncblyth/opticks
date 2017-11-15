#pragma once


class G4Step ; 

/**
CStep
======

* *CStep* ctor copies the argument G4Step 
  and holds the pointer to the copy together with step_id 

**/

#include "CFG4_API_EXPORT.hh"
class CFG4_API CStep {
   public:
       static double PreGlobalTime(const G4Step* step);
       static double PostGlobalTime(const G4Step* step);

       CStep(const G4Step* step, unsigned int step_id);
       virtual ~CStep();
       const G4Step* getStep() const ;  
       unsigned int  getStepId() const ; 
   private:
       const G4Step* m_step ; 
       unsigned int  m_step_id ; 
};


