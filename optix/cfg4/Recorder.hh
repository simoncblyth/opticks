#pragma once

#include "G4OpBoundaryProcess.hh"
#include <climits>
#include <cstring>

class G4Run ;
class G4Step ; 

template <typename T> class NPY ;
#include "RecorderBase.hh"

class Recorder : public RecorderBase {
   public:
        Recorder(const char* typ, const char* tag, const char* det, unsigned int record_max, unsigned int steps_per_photon, unsigned int photons_per_event);
        unsigned int getRecordMax();
        unsigned int getStepsPerPhoton();
        unsigned int getPhotonsPerEvent();
   public:
        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);
        void RecordStep(const G4Step*);
   public:
        void RecordStepPoint(const G4StepPoint* point, unsigned int record_id, unsigned int slot, unsigned int flag, bool last);
        void save();
   public:
        unsigned int getRecordId();
        unsigned int getEventId();
        unsigned int getPhotonId();
        unsigned int getStepId();
   public:
        unsigned int getPointFlag(G4StepPoint* point);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status);
        G4OpBoundaryProcessStatus getBoundaryStatus();
   private:
        void setEventId(unsigned int event_id);
        void setPhotonId(unsigned int photon_id);
        void setStepId(unsigned int step_id);
   private:
        void init();
   private:
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        unsigned int m_gen ; 
        unsigned int m_record_max ; 
        unsigned int m_steps_per_photon ; 
        unsigned int m_photons_per_event ; 

        unsigned int m_event_id ; 
        unsigned int m_photon_id ; 
        unsigned int m_step_id ; 
        G4OpBoundaryProcessStatus m_boundary_status ; 

        unsigned long long m_seqhis ; 
        unsigned long long m_seqmat ; 

        NPY<float>*               m_records ; 
        NPY<unsigned long long>*  m_history ; 

};

inline Recorder::Recorder(const char* typ, const char* tag, const char* det,unsigned int record_max, unsigned int steps_per_photon, unsigned int photons_per_event) 
   :
   m_typ(strdup(typ)),
   m_tag(strdup(tag)),
   m_det(strdup(det)),
   m_gen(0),
   m_record_max(record_max),
   m_steps_per_photon(steps_per_photon),
   m_photons_per_event(photons_per_event),
   m_event_id(UINT_MAX),
   m_photon_id(UINT_MAX),
   m_step_id(UINT_MAX),
   m_boundary_status(Undefined),
   m_seqhis(0),
   m_seqmat(0),
   m_records(0),
   m_history(0)
{
   init();
}


inline unsigned int Recorder::getRecordMax()
{
   return m_record_max ; 
}
inline unsigned int Recorder::getPhotonsPerEvent()
{
   return m_photons_per_event ; 
}
inline unsigned int Recorder::getStepsPerPhoton()
{
   return m_steps_per_photon ; 
}


inline unsigned int Recorder::getEventId()
{
   return m_event_id ; 
}
inline unsigned int Recorder::getPhotonId()
{
   return m_photon_id ; 
}
inline unsigned int Recorder::getStepId()
{
   return m_step_id ; 
}
inline unsigned int Recorder::getRecordId()
{
    return m_photons_per_event*m_event_id + m_photon_id ; 
}

inline G4OpBoundaryProcessStatus Recorder::getBoundaryStatus()
{
   return m_boundary_status ; 
}



inline void Recorder::setEventId(unsigned int event_id)
{
    m_event_id = event_id ; 
}
inline void Recorder::setPhotonId(unsigned int photon_id)
{
    m_photon_id = photon_id ; 
}
inline void Recorder::setStepId(unsigned int step_id)
{
    m_step_id = step_id ; 
}
inline void Recorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status)
{
   m_boundary_status = boundary_status ; 
}



inline void Recorder::RecordBeginOfRun(const G4Run*)
{
}

inline void Recorder::RecordEndOfRun(const G4Run*)
{
}


