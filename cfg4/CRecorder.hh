#pragma once

#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <glm/fwd.hpp>

// g4-
class G4Run ;
class G4Step ; 
class G4PrimaryVertex ; 
#include "G4ThreeVector.hh"


#include "CFG4_PUSH.hh"

#include "CBoundaryProcess.hh"
#include "CStage.hh"

#include "CFG4_POP.hh"

class Opticks ; // okc-
class OpticksEvent ; 


// cfg4-
class CRec ; 
class CGeometry ; 
class CMaterialBridge ; 
class CStp ; 

#include "CRecorder.h"

// npy-
template <typename T> class NPY ;


//  CRecorder
//  =============
//
//  The principal objective of *CRecorder* is to collect  
//  Geant4 photon steps in a format that precisely matches the
//  Opticks GPU photon records allowing use of the Opticks analysis 
//  and visualization tools.
//  To this end *CRecorder* saves non-dynamically into buffer of
//  fixed number of photons and max steps per photon 
//  in order to match on-GPU restrictions.  setQuad with
//  a computed record_id and slot_id is used to mimick
//  separate CUDA thread writes into tranches of record buffer. 
//
//  
//
//  CRecorder should really be called "OpticalPhotonCRecorder".
//  It is instanciated by CG4::configureGenerator 
//  and is mainly used from CSteppingAction.
//  It is also used for CRecorder::RecordPrimaryVertex.
//  from CGunSource and CTorchSource.
//
//  *RecordStep* is called for all G4Step
//  each of which is comprised of *pre* and *post* G4StepPoint, 
//  as a result the same G4StepPoint are "seen" twice, 
//  thus *RecordStep* only records the 1st of the pair 
//  (the 2nd will come around as the first at the next call)
//  except for the last G4Step pair where both points are recorded
//
//  *photons_per_g4event* is used by defineRecordId so the different
//  technical g4 events all get slotted into the same OpticksEvent record 
//  buffers
//
//
//
//  Traditional GPU Opticks simulation workflow:
//
//  * gensteps (Cerenkov/Scintillation) harvested from Geant4
//    and persisted into OpticksEvent
//
//  * gensteps seeded onto GPU using Thrust, summation over photons 
//    to generate per step provide photon and record buffer 
//    dimensions up frount 
//
//  * Cerenkov/Scintillation on GPU generation and propagation      
//    populate the pre-sized GPU record buffer 
//
//  This works because all gensteps are available before doing 
//  any optical simulation. BUT when operating on CPU doing the 
//  non-optical and optical simulation together, do not know the 
//  photon counts ahead of time.
//
//

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorder {
        friend class CSteppingAction ;
   public:
        static const char* PRE ; 
        static const char* POST ; 
        enum 
        {
           PRE_SAVE         = 0x1 << 0,
           POST_SAVE        = 0x1 << 1,
           PRE_DONE         = 0x1 << 2,
           POST_DONE        = 0x1 << 3,
           LAST_POST        = 0x1 << 4,
           SURF_ABS         = 0x1 << 5,
           PRE_SKIP         = 0x1 << 6,
           MAT_SWAP         = 0x1 << 7,
           STEP_START       = 0x1 << 8,
           STEP_REJOIN      = 0x1 << 9,
           STEP_RECOLL      = 0x1 << 10,
           RECORD_TRUNCATE  = 0x1 << 11,
           BOUNCE_TRUNCATE  = 0x1 << 12,
           ZERO_FLAG        = 0x1 << 13,
           DECREMENT_DENIED = 0x1 << 14,
           HARD_TRUNCATE    = 0x1 << 15,
           TOPSLOT_REWRITE  = 0x1 << 16
        };

        static const char* PRE_SAVE_ ; 
        static const char* POST_SAVE_ ; 
        static const char* PRE_DONE_ ; 
        static const char* POST_DONE_ ; 
        static const char* LAST_POST_ ; 
        static const char* SURF_ABS_ ; 
        static const char* PRE_SKIP_ ; 
        static const char* MAT_SWAP_ ; 
        static const char* STEP_START_ ; 
        static const char* STEP_REJOIN_ ; 
        static const char* STEP_RECOLL_ ; 
        static const char* RECORD_TRUNCATE_ ; 
        static const char* BOUNCE_TRUNCATE_ ; 
        static const char* HARD_TRUNCATE_ ; 
        static const char* ZERO_FLAG_ ; 
        static const char* DECREMENT_DENIED_ ; 
        static const char* TOPSLOT_REWRITE_ ; 

        static std::string Action(int action);
   public:
        CRecorder(Opticks* ok, CGeometry* geometry, bool dynamic);
        void postinitialize();  // called after G4 geometry constructed by CG4::postinitialize
        void initEvent(OpticksEvent* evt);      // MUST to be called prior to recording 
   public:
        // controlled via --dindex and --oindex options
        bool isDebug(); 
        bool isOther(); 
   private:
        void setDebug(bool debug);
        void setOther(bool other);
   private:
        void setEvent(OpticksEvent* evt);
   public:
        void posttrack(); 
        // formerly called from CSteppingAction::UserSteppingActionOptical on getting a new photon step before overwriting prior photon values 
        // now moved to CTrackingAction::PostUserTrackingAction
        void lookback(); 
   public:
        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);
        static double PreGlobalTime(const G4Step* step);
        static double PostGlobalTime(const G4Step* step);

        void RecordPhoton(const G4StepPoint* point); // overwrites target_record_id (ie m_record_id for static) entry for REJOINs
        void startPhoton();

   public:
        unsigned int getVerbosity();
   public:

#ifdef USE_CUSTOM_BOUNDARY
    public:
        bool Record(const G4Step* step, int step_id, int record_id, bool dbg, bool other, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label);
        void Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, 
                        unsigned mskhis, unsigned long long seqhis, unsigned long long seqmat, double time);
        void setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
        DsG4OpBoundaryProcessStatus getBoundaryStatus();
        void dump(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname );
#else
    public:
        bool Record(const G4Step* step, int step_id, int record_id, bool dbg, bool other, G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
        void Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, 
                        unsigned mskhis, unsigned long long seqhis, unsigned long long seqmat, double time);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
        G4OpBoundaryProcessStatus getBoundaryStatus();
        void dump(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname );
#endif
    private:
        void setStep(const G4Step* step, int step_id);
        bool RecordStep();
        void RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label);
        void RecordQuadrant();

        void Clear();
        bool hasIssue();
   public:
        std::string getStepActionString();
        bool isSelected(); 
        bool isHistorySelected(); 
        bool isMaterialSelected(); 
   public:
        // for reemission continuation
        void setSlot(unsigned slot);
        unsigned getSlot();
        void decrementSlot();
   public:
        // non-live running 
        void writeStps();
   public:
        void setEventId(int event_id);
        void setPhotonId(int photon_id);
        void setParentId(int parent_id);
        void setRecordId(int record_id, bool dbg, bool other);
        void setPrimaryId(int primary_id);
        void setStage(CStage::CStage_t stage);
   public:
        int getEventId();
        int getPhotonId();
        int getPhotonIdPrior();
        int getParentId();
        int getStepId();
        int getRecordId();

        unsigned getRecordMax();

        unsigned long long getSeqHis();
        unsigned long long getSeqMat();

        std::string description();

   public:
        // debugging/dumping 
        void Summary(const char* msg);
        void report(const char* msg="CRecorder::report");
        void dump(const char* msg="CRecorder::dump");
        void dump_full(const char* msg="CRecorder::dump_full");
        void addSeqhisMismatch(unsigned long long rdr, unsigned long long rec);
        void addSeqmatMismatch(unsigned long long rdr, unsigned long long rec);
        void addDebugPhoton(int photon_id);
   private:
        void init();
   private:
        Opticks*           m_ok; 
        unsigned long long m_dbgseqhis ;
        unsigned long long m_dbgseqmat ;
        bool               m_dbgflags ;

        CRec*              m_crec ; 
        OpticksEvent*      m_evt ; 
        CGeometry*         m_geometry ; 
        CMaterialBridge*   m_material_bridge ; 
        bool               m_dynamic ;
        bool               m_live ;   

        // m_live = true  : live recording mode, OpticksEvent records written during stepping
        // m_live = false : canned recording mode, records written Trajectory style from saved CRec CStp vector  

        unsigned int       m_gen ; 
       
        unsigned m_record_max ; 
        unsigned m_bounce_max ; 
        unsigned m_steps_per_photon ; 

        unsigned m_verbosity ; 
        bool     m_debug ; 
        bool     m_other ; 

        CStage::CStage_t m_stage ;
        CStage::CStage_t m_prior_stage ;

        int m_event_id ; 
        int m_photon_id ; 
        int m_photon_id_prior ; 
        int m_parent_id ; 
        int m_step_id ; 
        int m_record_id ; 
        int m_record_id_prior ; 
        int m_primary_id ; 



        uifchar4     m_c4 ; 


#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus m_boundary_status ; 
        DsG4OpBoundaryProcessStatus m_prior_boundary_status ; 
        std::vector<DsG4OpBoundaryProcessStatus>  m_bndstats ; 
#else
        G4OpBoundaryProcessStatus m_boundary_status ; 
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 
        std::vector<G4OpBoundaryProcessStatus>  m_bndstats ; 
#endif

        unsigned int m_premat ; 
        unsigned int m_prior_premat ; 

        unsigned int m_postmat ; 
        unsigned int m_prior_postmat ; 


        unsigned long long m_seqhis ; 
        unsigned long long m_seqmat ; 
        unsigned           m_mskhis ; 

        unsigned long long m_seqhis_select ; 
        unsigned long long m_seqmat_select ; 
        unsigned int       m_slot ; 
        unsigned int       m_decrement_request ; 
        unsigned int       m_decrement_denied ; 
        bool               m_record_truncate ; 
        bool               m_bounce_truncate ; 
        unsigned int       m_topslot_rewrite ; 
        unsigned           m_badflag ; 
        const G4Step*      m_step ; 
        unsigned           m_step_action ; 

        NPY<float>*               m_primary ; 
        NPY<float>*               m_photons ; 
        NPY<short>*               m_records ; 
        NPY<unsigned long long>*  m_history ; 


        NPY<short>*               m_dynamic_records ; 
        NPY<float>*               m_dynamic_photons ; 
        NPY<unsigned long long>*  m_dynamic_history ; 

        std::vector<const G4StepPoint*>         m_points ; 
        std::vector<unsigned int>               m_flags ; 
        std::vector<unsigned int>               m_materials ; 
        std::vector<unsigned long long>         m_seqhis_dbg  ; 
        std::vector<unsigned long long>         m_seqmat_dbg  ; 
        std::vector<unsigned>                   m_mskhis_dbg  ; 
        std::vector<double>                     m_times  ; 

        std::vector<std::pair<unsigned long long, unsigned long long> > m_seqhis_mismatch ; 
        std::vector<std::pair<unsigned long long, unsigned long long> > m_seqmat_mismatch ; 
        std::vector<int> m_debug_photon ; 


};
#include "CFG4_TAIL.hh"

