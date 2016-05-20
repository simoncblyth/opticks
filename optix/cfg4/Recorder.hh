#pragma once

#include <climits>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

// g4-

#include "G4OpBoundaryProcess.hh"
class G4Run ;
class G4Step ; 
class G4PrimaryVertex ; 

// cfg4-
class CPropLib ; 

#include "Recorder.h"

// npy-
#include "NumpyEvt.hpp"
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
        static const char* PRE ; 
        static const char* POST ; 
   public:
        Recorder(NumpyEvt* evt, unsigned int photons_per_g4event, unsigned int verbosity);
   public:
        void setPropLib(CPropLib* lib);
        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);
        bool RecordStep(const G4Step*);

        void RecordPhoton(const G4Step* step);
        void startPhoton();

        void setupPrimaryRecording();
        void RecordPrimaryVertex(G4PrimaryVertex* vertex);

        void Summary(const char* msg);
        void DumpSteps(const char* msg="Recorder::DumpSteps");
        void DumpStep(const G4Step* step);
   public:
        NumpyEvt* getEvt();
        unsigned int getVerbosity();
   public:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
        void RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label);
        void RecordQuadrant(const G4Step* step);

        void Clear();
        void Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis, unsigned long long seqmat);
        bool hasIssue();
   public:
        bool isSelected(); 
        bool isHistorySelected(); 
        bool isMaterialSelected(); 

        void Dump(const char* msg="Recorder::Dump");
        void Dump(const char* msg, unsigned int index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const char* matname );
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
        unsigned int getRecordMax();

        unsigned long long getSeqHis();
        unsigned long long getSeqMat();
   public:
        //unsigned int getPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
        G4OpBoundaryProcessStatus getBoundaryStatus();
   private:
        void init();
   private:
        NumpyEvt*    m_evt ; 
        CPropLib*    m_clib ; 

        unsigned int m_gen ; 

        unsigned int m_record_max ; 
        unsigned int m_bounce_max ; 
        unsigned int m_steps_per_photon ; 

        unsigned int m_photons_per_g4event ; 
        unsigned int m_verbosity ; 
        bool m_debug ; 
        unsigned int m_event_id ; 
        unsigned int m_photon_id ; 
        unsigned int m_step_id ; 
        unsigned int m_record_id ; 
        unsigned int m_primary_id ; 
        unsigned int m_primary_max ; 

        uifchar4     m_c4 ; 


        G4OpBoundaryProcessStatus m_boundary_status ; 
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 

        unsigned int m_premat ; 
        unsigned int m_prior_premat ; 

        unsigned int m_postmat ; 
        unsigned int m_prior_postmat ; 


        unsigned long long m_seqhis ; 
        unsigned long long m_seqmat ; 
        unsigned long long m_seqhis_select ; 
        unsigned long long m_seqmat_select ; 
        unsigned int       m_slot ; 
        bool               m_truncate ; 
        bool               m_step ; 

        NPY<float>*               m_primary ; 
        NPY<float>*               m_photons ; 
        NPY<short>*               m_records ; 
        NPY<unsigned long long>*  m_history ; 

        std::vector<const G4StepPoint*>         m_points ; 
        std::vector<unsigned int>               m_flags ; 
        std::vector<unsigned int>               m_materials ; 
        std::vector<G4OpBoundaryProcessStatus>  m_bndstats ; 
        std::vector<unsigned long long>         m_seqhis_dbg  ; 
        std::vector<unsigned long long>         m_seqmat_dbg  ; 


};

inline Recorder::Recorder(NumpyEvt* evt, unsigned int photons_per_g4event, unsigned int verbosity) 
   :
   m_evt(evt),
   m_clib(NULL),
   m_gen(0),

   m_record_max(0),
   m_bounce_max(0),
   m_steps_per_photon(0), 

   m_photons_per_g4event(photons_per_g4event),
   m_verbosity(verbosity),
   m_debug(verbosity > 0),
   m_event_id(UINT_MAX),
   m_photon_id(UINT_MAX),
   m_step_id(UINT_MAX),
   m_record_id(UINT_MAX),

   m_primary_id(UINT_MAX),
   m_primary_max(0),

   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),

   m_premat(0),
   m_prior_premat(0),

   m_postmat(0),
   m_prior_postmat(0),

   m_seqhis(0),
   m_seqmat(0),
   m_seqhis_select(0),
   m_seqmat_select(0),
   m_slot(0),
   m_truncate(false),
   m_step(true),

   m_primary(0),
   m_photons(0),
   m_records(0),
   m_history(0)
{
   init();
   
}


inline unsigned int Recorder::getVerbosity()
{
    return m_verbosity ; 
}
inline bool Recorder::isHistorySelected()
{
   return m_seqhis_select == m_seqhis ; 
}
inline bool Recorder::isMaterialSelected()
{
   return m_seqmat_select == m_seqmat ; 
}
inline bool Recorder::isSelected()
{
   return isHistorySelected() || isMaterialSelected() ;
}

inline unsigned long long Recorder::getSeqHis()
{
    return m_seqhis ; 
}
inline unsigned long long Recorder::getSeqMat()
{
    return m_seqmat ; 
}






inline void Recorder::setPropLib(CPropLib* clib)
{
    m_clib = clib  ; 
}


inline NumpyEvt* Recorder::getEvt()
{
    return m_evt ; 
}
inline unsigned int Recorder::getRecordMax()
{
    return m_record_max ; 
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


