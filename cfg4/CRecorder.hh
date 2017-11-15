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

struct CG4Ctx ; 
#include "CPhoton.hh"

class CRec ; 
class CGeometry ; 
class CMaterialBridge ; 
class CWriter ; 
class CStp ; 

#include "CRecorder.h"

// npy-
template <typename T> class NPY ;


/**

CRecorder
=============


TODO
~~~~~

1. perhaps split into two CLiveRecorder and CCannedRecorder ?
2. factor off the dumping into separate class 

3. is there any way to structure CRecorder similar to oxrap/cu/generate.cu ?
    which is what it is trying to emulate afterall 


The principal objective of *CRecorder* is to collect  
Geant4 photon steps in a format that precisely matches the
Opticks GPU photon records allowing use of the Opticks analysis 
and visualization tools.
To this end *CRecorder* saves non-dynamically into buffer of
fixed number of photons and max steps per photon 
in order to match on-GPU restrictions.  setQuad with
a computed record_id and slot_id is used to mimick
separate CUDA thread writes into tranches of record buffer. 

CRecorder should really be called "OpticalPhotonCRecorder".
It is instanciated by CG4::configureGenerator 
and is mainly used from CSteppingAction.
It is also used for CRecorder::RecordPrimaryVertex.
from CGunSource and CTorchSource.




*LiveRecordStep* is called for all G4Step
each of which is comprised of *pre* and *post* G4StepPoint, 
as a result the same G4StepPoint are "seen" twice, 
thus *RecordStep* only records the 1st of the pair 
(the 2nd will come around as the first at the next call)
except for the last G4Step pair where both points are recorded

*photons_per_g4event* is used by defineRecordId so the different
technical g4 events all get slotted into the same OpticksEvent record 
buffers


Traditional GPU Opticks simulation workflow:

* gensteps (Cerenkov/Scintillation) harvested from Geant4
  and persisted into OpticksEvent

* gensteps seeded onto GPU using Thrust, summation over photons 
  to generate per step provide photon and record buffer 
  dimensions up frount 

* Cerenkov/Scintillation on GPU generation and propagation      
  populate the pre-sized GPU record buffer 

This works because all gensteps are available before doing 
any optical simulation. BUT when operating on CPU doing the 
non-optical and optical simulation together, do not know the 
photon counts ahead of time.

**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorder {
        friend class CSteppingAction ;
   public:
        static const char* PRE ; 
        static const char* POST ; 
   public:
        CRecorder(CG4* g4, CGeometry* geometry, bool dynamic);
        std::string desc() const ; 
        void postinitialize();  // called after G4 geometry constructed by CG4::postinitialize
        void initEvent(OpticksEvent* evt);      // MUST to be called prior to recording 
   private:
        void setEvent(OpticksEvent* evt);
   public:
        void posttrack(); // invoked from CTrackingAction::PostUserTrackingAction
   public:
        void RecordBeginOfRun(const G4Run*);
        void RecordEndOfRun(const G4Run*);

        void startPhoton();

   public:
        unsigned int getVerbosity();
   public:

#ifdef USE_CUSTOM_BOUNDARY
    public:
        bool Record(DsG4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label);
        void Collect(const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, const CPhoton& photon );
        void setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
        DsG4OpBoundaryProcessStatus getBoundaryStatus();
        void dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname );
#else
    public:
        bool Record(G4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
        void Collect(const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const CPhoton& photon );
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
        G4OpBoundaryProcessStatus getBoundaryStatus();
        void dump_point(const G4ThreeVector& origin, unsigned index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, unsigned flag, const char* matname );
#endif
    private:
        void setStep(const G4Step* step, int step_id);
        //bool LiveRecordStep();
        void RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label);
        void RecordQuadrant(uifchar4& c4);

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
        void CannedWriteSteps();
   public:
        unsigned long long getSeqHis();
        unsigned long long getSeqMat();

        std::string description();

   public:
        // debugging/dumping 
        void Summary(const char* msg);
        void dump(const char* msg="CRecorder::dump");
        void dump_brief(const char* msg="CRecorder::dump_brief");
        void dump_sequence(const char* msg="CRecorder::dump_sequence");
        void dump_points(const char* msg="CRecorder::dump_points");
        void dumpStepVelocity(const char* msg="CRecorder::dumpStepVelocity");
   public:
        // reporting
        void report(const char* msg="CRecorder::report");
        void addSeqhisMismatch(unsigned long long rdr, unsigned long long rec);
        void addSeqmatMismatch(unsigned long long rdr, unsigned long long rec);
        void addDebugPhoton(int photon_id);
   private:
        void init();
   private:
        CG4*               m_g4; 
        CG4Ctx&            m_ctx; 
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
        CWriter*           m_writer ; 

        // m_live = true  : live recording mode, OpticksEvent records written during stepping
        // m_live = false : canned recording mode, records written Trajectory style from saved CRec CStp vector  


        unsigned m_verbosity ; 



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


        CPhoton     m_photon ;  
        unsigned long long m_seqhis_select ; 
        unsigned long long m_seqmat_select ; 
        unsigned int       m_slot ; 
        unsigned int       m_decrement_request ; 
        unsigned int       m_decrement_denied ; 
        bool               m_record_truncate ; 
        bool               m_bounce_truncate ; 
        unsigned int       m_topslot_rewrite ; 
        unsigned           m_badflag ; 
        //const G4Step*      m_step ; 
        unsigned           m_step_action ; 

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

