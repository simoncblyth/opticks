#pragma once

#include <vector>

// g4
class G4Step ; 
class G4StepPoint ; 

// cg4-
class CStep ;

// npy-
class NumpyEvt ; 
template <typename T> class NPY ;


class CStepRec {
   public:
       CStepRec( NumpyEvt* evt );
   private:
       void init();    
   public:
       void record(const G4Step* step, unsigned int step_id);
       void store(unsigned int event_id, unsigned int track_id, int particle_id);
       void store(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point);
   private:
       NumpyEvt*                   m_evt ; 
       NPY<float>*                 m_nopstep ; 
       std::vector<const CStep*>  m_steps ; 

};


inline CStepRec::CStepRec( NumpyEvt* evt )
   :
   m_evt(evt),
   m_nopstep(NULL)
{
    init();
}
