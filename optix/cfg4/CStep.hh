#pragma once

class G4Step ; 

class CStep {
   public:
       CStep(const G4Step* step, unsigned int step_id);
       virtual ~CStep();
       const G4Step* getStep() const ;  
       unsigned int  getStepId() const ; 
   private:
       const G4Step* m_step ; 
       unsigned int  m_step_id ; 
};


inline const G4Step* CStep::getStep() const 
{
    return m_step ; 
}
inline unsigned int CStep::getStepId() const 
{
    return m_step_id ; 
}


