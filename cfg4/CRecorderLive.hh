#pragma once

// **CURRENTLY DEAD CODE**

// fork(or have both for crosscheck) at higher CG4 level 
//   m_live(m_ok->hasOpt("liverecorder")),



#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorderLive {
        friend class CSteppingAction ;
   public:
        static const char* PRE ; 
        static const char* POST ; 
   public:
        CRecorderLive(CG4* g4, CGeometry* geometry, bool dynamic);
        void postinitialize();               // called after G4 geometry constructed in CG4::postinitialize
        void initEvent(OpticksEvent* evt);   // called prior to recording, sets up writer (output buffers)
        CRec* getCRec() const ; 
   private:
        void setEvent(OpticksEvent* evt);
   public:
        void posttrack();                    // invoked from CTrackingAction::PostUserTrackingAction for optical photons
   public:
        void zeroPhoton();
   public:

#ifdef USE_CUSTOM_BOUNDARY
    public:
        bool Record(DsG4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label);
        void setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
#else
    public:
        bool Record(G4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
#endif
    private:
        void setStep(const G4Step* step, int step_id);
        //bool LiveRecordStep();
        void RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label);
        //void Clear();
   public:
        std::string getStepActionString();
   public:
        // for reemission continuation
        void decrementSlot();
   public:
        void posttrackWriteSteps();
   public:
        void Summary(const char* msg);
        std::string desc() const ; 
        void dump(const char* msg="CRecorder::dump");
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


    private:


        // below are zeroed in zeroPhoton
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus m_boundary_status ; 
        DsG4OpBoundaryProcessStatus m_prior_boundary_status ; 
#else
        G4OpBoundaryProcessStatus m_boundary_status ; 
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 
#endif
        unsigned m_premat ; 
        unsigned m_prior_premat ; 

        unsigned m_postmat ; 
        unsigned m_prior_postmat ; 





};
#include "CFG4_TAIL.hh"

