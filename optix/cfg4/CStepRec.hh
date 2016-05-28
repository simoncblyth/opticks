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
       CStepRec(NumpyEvt* evt);
   private:
       void init();    
   public:
       void collectStep(const G4Step* step, unsigned int step_id);   
       void storeStepsCollected(unsigned int event_id, unsigned int track_id, int particle_id);
       unsigned int getStoreCount();
   private:
       void storePoint(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point);
   private:
       NumpyEvt*                   m_evt ; 
       NPY<float>*                 m_nopstep ; 
       std::vector<const CStep*>   m_steps ; 
       unsigned int                m_store_count ; 

};


inline CStepRec::CStepRec( NumpyEvt* evt )
   :
   m_evt(evt),
   m_nopstep(NULL),
   m_store_count(0)
{
    init();
}

inline unsigned int CStepRec::getStoreCount()
{
   return m_store_count ; 
}

