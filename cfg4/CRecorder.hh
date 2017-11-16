#pragma once

#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <glm/fwd.hpp>

// g4-
//class G4Run ;
class G4Step ; 
//class G4PrimaryVertex ; 

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
#include "CRecState.hh"

class CDebug ; 
class CRec ; 
class CGeometry ; 
class CMaterialBridge ; 
class CWriter ; 
class CStp ; 


/**

CRecorder
=============

TODO 
   Despite extensive refactoring (breaking up the old monolith)
   CRecorder is still confusing... because the old live recording code
   is still mixed up with the canned recording : and its difficult
   to disentangle which is which.



Canonical m_recorder instance is resident of CG4 and is
instanciated with it.
CRecorder should really be called "OpticalPhotonCRecorder".
It is mainly used from CSteppingAction, via the *Record* method. 

The objective of *CRecorder* is to collect Geant4 photon 
steps in a format that precisely matches the Opticks GPU 
photon records allowing use of the Opticks analysis 
and visualization tools with G4 simulations.

To this end *CRecorder* saves non-dynamically into buffer of
fixed number of photons and max steps per photon 
in order to match on-GPU restrictions.  setQuad with
a computed record_id and slot_id is used to mimick
separate CUDA thread writes into tranches of record buffer. 

Modes of operation:

*canned*
     canned recording mode, records written Trajectory style from saved CRec CStp vector  

*live*

canned mode
-------------


live recording mode : **currently not used**
---------------------------------------------

* much of the code for this currently parked in CRecorderDead.cc

OpticksEvent records written during stepping.
*LiveRecordStep* is called for all G4Step
each of which is comprised of *pre* and *post* G4StepPoint, 
as a result the same G4StepPoint are "seen" twice, 
thus *RecordStep* only records the 1st of the pair 
(the 2nd will come around as the first at the next call)
except for the last G4Step pair where both points are recorded

*photons_per_g4event* is used by defineRecordId so the different
technical g4 events all get slotted into the same OpticksEvent record 
buffers



Example
---------

Consider 
   TO RE BT BT BT BT SA

Live mode:
   write pre until last when write pre,post 

Canned mode:
    For first write pre,post then write post

Rejoins are not known until another track comes along 
that lines up with former ending in AB. 

**/

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorder {
        friend class CSteppingAction ;
   public:
        static const char* PRE ; 
        static const char* POST ; 
   public:
        std::string getStepActionString();
        CRec*       getCRec() const ; 
   public:
        CRecorder(CG4* g4, CGeometry* geometry, bool dynamic); // CG4::CG4
        void postinitialize();               // called after G4 geometry constructed in CG4::postinitialize
        void initEvent(OpticksEvent* evt);   // called prior to recording, sets up writer (output buffers)
        void posttrack();                    // invoked from CTrackingAction::PostUserTrackingAction for optical photons
#ifdef USE_CUSTOM_BOUNDARY
        bool Record(DsG4OpBoundaryProcessStatus boundary_status);
#else
        bool Record(G4OpBoundaryProcessStatus boundary_status);
#endif
   private:
        void zeroPhoton();
        void decrementSlot(); // for reemission continuation
        void posttrackWriteSteps();
 
#ifdef USE_CUSTOM_BOUNDARY
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label);
#else
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
#endif
  public:
        void Summary(const char* msg);
        void dump(const char* msg="CRecorder::dump");
        std::string desc() const ; 
   private:
        CG4*               m_g4; 
        CG4Ctx&            m_ctx; 
        Opticks*           m_ok; 
        CPhoton            m_photon ;  
        CRecState          m_state ;  
        CRec*              m_crec ; 
        CDebug*            m_dbg ; 

        OpticksEvent*      m_evt ; 
        CGeometry*         m_geometry ; 
        CMaterialBridge*   m_material_bridge ; 
        bool               m_dynamic ;
        bool               m_live ;   
        CWriter*           m_writer ; 

};
#include "CFG4_TAIL.hh"

