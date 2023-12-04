#pragma once
#include <string>
#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"
struct NP ; 

/**
SEventConfig
==============

This is for configuration that is important and/or where 
the settings have impact across multiple classes/structs. 

Note that geometry related things are configured in SGeoConfig. 

For settings that are local to individual structs or mainly for 
shortterm debug : **do not implement here**. Instead 
implement local config locally using constexpr envvar keys 
inside the structs. 

Primary user of this config is QEvent::init 


EventMode
    configures what will be persisted, ie what is in the SEvt
RunningMode
    configures how running is done, eg Default/DefaultSaveG4State/RerunG4State 

MaxPhoton

MaxSimtrace

MaxCurandState
   from std::max of MaxPhoton and MaxSimtrace

MaxRec
    normally 0, disabling creation of the QEvent domain compressed step record buffer

MaxRecord 
    normally 0, disabling creation of the QEvent full photon step record buffer.
    When greater than zero causes QEvent::setNumPhoton to allocate the full
    step record buffer, full recording requires a value of one greater 
    than the MaxBounce configured  (typically up to 16) 

MaxExtent (mm)
    only relevant to the domain compression of the compressed step records

MaxTime (ns)
    only relevant to the domain compression of the compressed step records


+-------------------------------+-----------------------------------------+---------------------------------------+
| Method                        |  Default                                | envvar                                |
+===============================+=========================================+=======================================+
| SEventConfig::OutFold         | "$DefaultOutputDir"                     | OPTICKS_OUT_FOLD                      |
+-------------------------------+-----------------------------------------+---------------------------------------+

**/


struct SYSRAP_API SEventConfig
{
    static const plog::Severity LEVEL ;  
    static const int LIMIT ; 
    static void Check(); 
    static std::string Desc(); 
    static std::string HitMaskLabel(); 


    // [TODO : RECONSIDER OUTDIR OUTNAME MECHANICS FOLLOWING SEVT LAYOUT 
    static const char* OutDir( const char* reldir); 
    static const char* OutPath( const char* reldir, const char* stem, int index, const char* ext ); 

    static const char* OutDir(); 
    static const char* OutPath( const char* stem, int index, const char* ext ); 
    static std::string DescOutPath(  const char* stem, int index, const char* ext ) ; 
    // ]TODO

    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static constexpr const char* kIntegrationMode = "OPTICKS_INTEGRATION_MODE" ; 
    static constexpr const char* kEventMode    = "OPTICKS_EVENT_MODE" ; 
    static constexpr const char* kRunningMode  = "OPTICKS_RUNNING_MODE" ; 
    static constexpr const char* kStartIndex   = "OPTICKS_START_INDEX" ; 
    static constexpr const char* kNumEvent     = "OPTICKS_NUM_EVENT" ; 
    static constexpr const char* kNumPhoton    = "OPTICKS_NUM_PHOTON" ; 
    static constexpr const char* kG4StateSpec  = "OPTICKS_G4STATE_SPEC" ; 
    static constexpr const char* kG4StateRerun = "OPTICKS_G4STATE_RERUN" ; 

    static constexpr const char* kMaxGenstep   = "OPTICKS_MAX_GENSTEP" ; 
    static constexpr const char* kMaxPhoton    = "OPTICKS_MAX_PHOTON" ; 
    static constexpr const char* kMaxSimtrace  = "OPTICKS_MAX_SIMTRACE" ; 

    static constexpr const char* kMaxBounce    = "OPTICKS_MAX_BOUNCE" ; 
    static constexpr const char* kMaxRecord    = "OPTICKS_MAX_RECORD" ; 
    static constexpr const char* kMaxRec       = "OPTICKS_MAX_REC" ; 
    static constexpr const char* kMaxAux       = "OPTICKS_MAX_AUX" ; 
    static constexpr const char* kMaxSup       = "OPTICKS_MAX_SUP" ; 
    static constexpr const char* kMaxSeq       = "OPTICKS_MAX_SEQ" ; 
    static constexpr const char* kMaxPrd       = "OPTICKS_MAX_PRD" ; 
    static constexpr const char* kMaxTag       = "OPTICKS_MAX_TAG" ; 
    static constexpr const char* kMaxFlat      = "OPTICKS_MAX_FLAT" ; 

    static constexpr const char* kMaxExtent    = "OPTICKS_MAX_EXTENT" ; 
    static constexpr const char* kMaxTime      = "OPTICKS_MAX_TIME" ; 

    static constexpr const char* kOutPrefix    = "OPTICKS_OUT_PREFIX" ; 
    static constexpr const char* kOutFold      = "OPTICKS_OUT_FOLD" ; 
    static constexpr const char* kOutName      = "OPTICKS_OUT_NAME" ; 
    static constexpr const char* kHitMask      = "OPTICKS_HIT_MASK" ; 
    static constexpr const char* kRGMode       = "OPTICKS_RG_MODE" ; 

    static constexpr const char* kGatherComp   = "OPTICKS_GATHER_COMP" ; 
    static constexpr const char* kSaveComp     = "OPTICKS_SAVE_COMP" ; 

    static constexpr const char* kPropagateEpsilon = "OPTICKS_PROPAGATE_EPSILON" ; 
    static constexpr const char* kInputGenstep     = "OPTICKS_INPUT_GENSTEP" ; 
    static constexpr const char* kInputPhoton      = "OPTICKS_INPUT_PHOTON" ; 
    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ; 


    static int         IntegrationMode(); 
    static bool        GPU_Simulation() ; // 1 or 3 
    static bool        CPU_Simulation() ; // 2 or 3 

    static const char* EventMode(); 

    static int         RunningMode(); 
    static const char* RunningModeLabel(); 

    static bool IsRunningModeDefault(); 
    static bool IsRunningModeG4StateSave(); 
    static bool IsRunningModeG4StateRerun();  

    static bool IsRunningModeTorch();  
    static bool IsRunningModeInputPhoton();  
    static bool IsRunningModeInputGenstep();  
    static bool IsRunningModeGun();  


    static const char* G4StateSpec(); 
    static int         G4StateRerun(); 

    static int MaxGenstep(); 
    static int MaxPhoton(); 
    static int MaxSimtrace(); 
    static int MaxCurandState();  // from max of MaxPhoton and MaxSimtrace

    static int MaxBounce(); 
    static int MaxRecord();  // full photon step record  
    static int MaxRec();     // compressed photon step record
    static int MaxAux();     // auxiliary photon step record, used for debug 
    static int MaxSup();     // supplementry photon level info
    static int MaxSeq();     // seqhis slots
    static int MaxPrd();    
    static int MaxTag();    
    static int MaxFlat();    
    static float MaxExtent() ; 
    static float MaxTime() ; 
    static const char* OutFold(); 
    static const char* OutName(); 
    static unsigned HitMask(); 

    static unsigned GatherComp(); 
    static unsigned SaveComp(); 

    static float PropagateEpsilon(); 
    static const char* InputGenstep(); 
    static const char* InputPhoton(); 
    static const char* InputPhotonFrame(); 

    static int RGMode(); 
    static bool IsRGModeRender(); 
    static bool IsRGModeSimtrace(); 
    static bool IsRGModeSimulate(); 
    static bool IsRGModeTest(); 

    static const char* RGModeLabel(); 

    static std::string GatherCompLabel(); 
    static std::string SaveCompLabel(); 

    static void GatherCompList( std::vector<unsigned>& gather_comp ) ; 
    static void SaveCompList( std::vector<unsigned>& save_comp ) ; 

    static constexpr const char* Default = "Default" ; 
    static constexpr const char* StandardFullDebug = "StandardFullDebug" ; 
    static constexpr const char* Minimal = "Minimal" ; 
    static constexpr const char* HitOnly = "HitOnly" ; 
    static constexpr const char* HitAndPhoton = "HitAndPhoton" ; 

    static void SetDefault(); 
    static void SetStandardFullDebug(); 
    static void SetMinimal();
    static void SetHitOnly();
    static void SetHitAndPhoton();

    static bool IsDefault(); 
    static bool IsStandardFullDebug(); 
    static bool IsMinimal();
    static bool IsHitOnly();
    static bool IsHitAndPhoton();

    static void SetIntegrationMode(int mode);   // IntegrationMode configures the integration of Opticks and Framework 
    static void SetEventMode(const char* mode);   // EventMode configures what will be persisted, ie what is in the SEvt
    static void SetRunningMode(const char* mode); // RunningMode configures how running is done, eg Default/DefaultSaveG4State/RerunG4State/Torch

    static void SetStartIndex(int index0); 
    static void SetNumEvent(int nevt);            // NumEvent is used by some tests 
    static void SetNumPhoton(const char* spec);   // NumPhoton is used by some tests 
    static void SetG4StateSpec(const char* spec); 
    static void SetG4StateRerun(int id); 

    static void SetMaxGenstep(int max_genstep); 
    static void SetMaxPhoton( int max_photon); 
    static void SetMaxSimtrace( int max_simtrace); 
    static void SetMaxBounce( int max_bounce); 
    static void SetMaxRecord( int max_record); 
    static void SetMaxRec(    int max_rec); 
    static void SetMaxAux(    int max_aux); 
    static void SetMaxSup(    int max_sup); 
    static void SetMaxSeq(    int max_seq); 
    static void SetMaxPrd(    int max_prd); 
    static void SetMaxTag(    int max_tag); 
    static void SetMaxFlat(    int max_flat); 

    static void SetMaxExtent( float max_extent); 
    static void SetMaxTime(   float max_time ); 

    static void SetOutFold( const char* out_fold); 
    static void SetOutName( const char* out_name); 
    static void SetHitMask(const char* abrseq, char delim=',' ); 

    static void SetRGMode( const char* rg_mode) ; 
    static void SetRGModeSimulate() ; 
    static void SetRGModeSimtrace() ; 
    static void SetRGModeRender() ; 
    static void SetRGModeTest() ; 

    static void SetPropagateEpsilon( float eps) ; 
    static void SetInputGenstep(const char* input_genstep); 
    static void SetInputPhoton(const char* input_photon); 
    static void SetInputPhotonFrame(const char* input_photon_frame); 

    static void SetGatherComp_(unsigned mask); 
    static void SetGatherComp(const char* names, char delim=',') ; 

    static void SetSaveComp_(unsigned mask); 
    static void SetSaveComp(const char* names, char delim=',') ; 

    static void  SetComp(); 
    static void  CompAuto(unsigned& gather_mask, unsigned& save_mask ); 


    static int         _IntegrationModeDefault ; 
    static const char* _EventModeDefault ; 
    static const char* _RunningModeDefault ; 
    static int         _StartIndexDefault ; 
    static int         _NumEventDefault ; 
    static const char* _NumPhotonDefault ; 
    static const char* _G4StateSpecDefault ; 
    static const char* _G4StateSpecNotes ; 
    static int         _G4StateRerunDefault ; 
    static const char* _MaxBounceNotes ; 

    static const char* _MaxGenstepDefault ; 
    static const char* _MaxPhotonDefault ; 
    static const char* _MaxSimtraceDefault ; 

    static int _MaxBounceDefault ; 
    static int _MaxRecordDefault ; 
    static int _MaxRecDefault ; 
    static int _MaxAuxDefault ; 
    static int _MaxSupDefault ; 
    static int _MaxSeqDefault ; 
    static int _MaxPrdDefault ; 
    static int _MaxTagDefault ; 
    static int _MaxFlatDefault ; 
    static float _MaxExtentDefault ; 
    static float _MaxTimeDefault  ; 
    static const char* _OutFoldDefault ; 
    static const char* _OutNameDefault ; 
    static const char* _HitMaskDefault ; 
    static const char* _RGModeDefault ; 

    static const char* _GatherCompDefault ; 
    static const char* _SaveCompDefault ; 

    static float _PropagateEpsilonDefault  ; 
    static const char* _InputGenstepDefault ; 
    static const char* _InputPhotonDefault ; 
    static const char* _InputPhotonFrameDefault ; 


    static int         _IntegrationMode ; 
    static const char* _EventMode ; 
    static int         _RunningMode ; 
    static int         _StartIndex ; 
    static int         _NumEvent ; 

    static std::vector<int>* _GetNumPhotonPerEvent(); 
    static std::vector<int>* _NumPhotonPerEvent ; 
    static int               _GetNumPhoton(int idx); 
    static int               _GetNumEvent(); 
    static int               NumPhoton(int idx); // some tests need varying photon count 
    static int               NumEvent();         // some tests use event count and nned to detect last event 
    static int               EventIndex(int idx) ; 
 

    static const char* _G4StateSpec ; 
    static int         _G4StateRerun ; 

    static int _MaxGenstep ; 
    static int _MaxPhoton ; 
    static int _MaxSimtrace ; 
    static int _MaxBounce ; 
    static int _MaxRecord ; 
    static int _MaxRec ; 
    static int _MaxAux ; 
    static int _MaxSup ; 
    static int _MaxSeq ; 
    static int _MaxPrd ; 
    static int _MaxTag ; 
    static int _MaxFlat ; 
    static float _MaxExtent ; 
    static float _MaxTime  ; 
    static const char* _OutFold ; 
    static const char* _OutName ; 
    static unsigned _HitMask ; 
    static int _RGMode ; 

    static unsigned _GatherComp ; 
    static unsigned _SaveComp ; 

    static float _PropagateEpsilon ;
    static const char* _InputGenstep ; 
    static const char* _InputPhoton ; 
    static const char* _InputPhotonFrame ; 


    static int Initialize_COUNT ; 
    static int Initialize(); 
    static uint64_t EstimateAlloc(); 

    static constexpr const char* NAME = "SEventConfig.npy" ; 
    static NP* Serialize(); 
    static void Save(const char* dir) ; 

}; 
 
