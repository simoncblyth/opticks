#pragma once

class G4Run ;
class G4Step ; 

template <typename T> class NPY ;
#include "RecorderBase.hh"

class Recorder : public RecorderBase {
   public:
        Recorder(unsigned int photon_max, unsigned int step_max);

        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);
        void RecordStep(const G4Step*);
        void save(const char*);
   private:
        void init();
   private:
        unsigned int m_photon_max ; 
        unsigned int m_step_max ; 
        NPY<float>*  m_recs ; 


};

inline Recorder::Recorder(unsigned int photon_max, unsigned int step_max) 
   :
   m_photon_max(photon_max),
   m_step_max(step_max),
   m_recs(0)
{
   init();
}

