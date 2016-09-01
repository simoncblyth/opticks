#pragma once

#include <vector>

// g4-
class G4Step ; 
class G4StepPoint ; 

class OpticksHub ; // okg-
class CStep ;     // cg4-


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/** 
CStepRec : Records non-optical particle steps 
================================================

Non-optical steps are serialized into the nopstep buffer of the 
current G4 event.  As the nopstep pointer is updated by 
OpticksHub as each G4 event is created this class needs
to take no action on changing event. It just keeps
recording into the nopstep provided by the hub.

**/


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
       std::vector<const CStep*>   m_steps ; 
       unsigned int                m_store_count ; 

};

#include "CFG4_TAIL.hh"


