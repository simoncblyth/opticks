#pragma once

#include "G4OpBoundaryProcess.hh"

#include <climits>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

class G4Run ;
class G4Step ; 

template <typename T> class NPY ;

//
//  *RecordStep* is called for all G4Step
//  each of which is comprised of *pre* and *post* G4StepPoint, 
//  as a result the same G4StepPoint are "seen" twice, 
//  thus *RecordStep* only records the 1st of the pair 
//  (the 2nd will come around as the first at the next call)
//  except for the last G4Step pair where both points are recorded
//

class Recorder {
   public:
        Recorder(const char* typ, const char* tag, const char* det, unsigned int record_max, unsigned int steps_per_photon, unsigned int photons_per_g4event);
        unsigned int getRecordMax();
        unsigned int getStepsPerPhoton();
   public:
        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);
        void RecordStep(const G4Step*);
        void startPhoton();

        void DumpSteps(const char* msg="Recorder::DumpSteps");
        void DumpStep(const G4Step* step);
   public:
        void RecordStepPoint(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status, bool last);
        void Clear();
        void Collect(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis);
        bool hasIssue();
   public:
        void setCenterExtent(const glm::vec4& center_extent);
        void setBoundaryDomain(const glm::vec4& boundary_domain);
        void Dump(const char* msg="Recorder::Dump");
        void save();
   public:
        void setEventId(unsigned int event_id);
        void setPhotonId(unsigned int photon_id);
        void setStepId(unsigned int step_id);
        void setRecordId(unsigned int record_id);
   public:
        unsigned int getEventId();
        unsigned int getPhotonId();
        unsigned int getStepId();
        unsigned int defineRecordId();
        unsigned int getRecordId();
   public:
        unsigned int getPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status);
        G4OpBoundaryProcessStatus getBoundaryStatus();
   private:
        void init();
   private:
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        unsigned int m_gen ; 
        unsigned int m_record_max ; 
        unsigned int m_steps_per_photon ; 
        unsigned int m_photons_per_g4event ; 

        unsigned int m_event_id ; 
        unsigned int m_photon_id ; 
        unsigned int m_step_id ; 
        unsigned int m_record_id ; 
        G4OpBoundaryProcessStatus m_boundary_status ; 
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 

        unsigned long long m_seqhis_select ; 
        unsigned long long m_seqhis ; 
        unsigned long long m_seqmat ; 
        unsigned int m_slot ; 

        NPY<float>*               m_fdom ; 
        NPY<int>*                 m_idom ; 
        NPY<float>*               m_photons ; 
        NPY<short>*               m_records ; 
        NPY<unsigned long long>*  m_history ; 

        glm::vec4    m_center_extent ; 
        glm::vec4    m_time_domain ; 
        glm::vec4    m_boundary_domain ; 

        std::vector<const G4StepPoint*>         m_points ; 
        std::vector<unsigned int>               m_flags ; 
        std::vector<G4OpBoundaryProcessStatus>  m_bndstats ; 
        std::vector<unsigned long long>         m_seqhis_dbg  ; 


};

inline Recorder::Recorder(const char* typ, const char* tag, const char* det, unsigned int record_max, unsigned int steps_per_photon, unsigned int photons_per_g4event) 
   :
   m_typ(strdup(typ)),
   m_tag(strdup(tag)),
   m_det(strdup(det)),
   m_gen(0),
   m_record_max(record_max),
   m_steps_per_photon(steps_per_photon),
   m_photons_per_g4event(photons_per_g4event),
   m_event_id(UINT_MAX),
   m_photon_id(UINT_MAX),
   m_step_id(UINT_MAX),
   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),
   m_seqhis(0),
   m_seqhis_select(0),
   m_seqmat(0),
   m_slot(0),
   m_fdom(0),
   m_idom(0),
   m_photons(0),
   m_records(0),
   m_history(0),
   m_center_extent(0.f,0.f,0.f,0.f),
   m_time_domain(0.f,0.f,0.f,0.f),
   m_boundary_domain(0.f,0.f,0.f,0.f)
{
   init();
}

inline unsigned int Recorder::getRecordMax()
{
   return m_record_max ; 
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
   return m_record_id ; 
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
inline unsigned int Recorder::defineRecordId()   
{
   return m_photons_per_g4event*m_event_id + m_photon_id ; 
}

inline void Recorder::setRecordId(unsigned int record_id)
{
    m_record_id = record_id ; 
}






inline void Recorder::RecordBeginOfRun(const G4Run*)
{
}

inline void Recorder::RecordEndOfRun(const G4Run*)
{
}

inline void Recorder::setCenterExtent(const glm::vec4& center_extent)
{
   m_center_extent = center_extent ; 
}
inline void Recorder::setBoundaryDomain(const glm::vec4& boundary_domain)
{
   m_boundary_domain = boundary_domain ; 
}


