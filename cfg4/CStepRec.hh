#pragma once

#include <vector>

// g4-
class G4Step ; 
class G4StepPoint ; 

// okc-
class OpticksEvent ; 

// opticksgeo-
class OpticksHub ; 


// cg4-
class CStep ;

// npy-
template <typename T> class NPY ;


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CStepRec {
   public:
       CStepRec(OpticksHub* hub);
   private:
       void init();    
   public:
       void collectStep(const G4Step* step, unsigned int step_id);   
       void storeStepsCollected(unsigned int event_id, unsigned int track_id, int particle_id);
       unsigned int getStoreCount();
   private:
       void storePoint(unsigned int event_id, unsigned int track_id, int particle_id, unsigned int point_id, const G4StepPoint* point);
   private:
       OpticksHub*                 m_hub ; 
       OpticksEvent*               m_evt ; 
       NPY<float>*                 m_nopstep ; 
       std::vector<const CStep*>   m_steps ; 
       unsigned int                m_store_count ; 

};

#include "CFG4_TAIL.hh"


