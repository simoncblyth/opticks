#pragma once
#include <string>
#include <vector>
#include <limits>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct NP ; 
struct scontext ; 
struct salloc ; 

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


Usage
------

Usage is widespread, some of the more pertinent places: 

SEvt::SEvt
   invokes SEventConfig::Initialize
   (HMM: run twice for for A-B testing when have multiple SEvt)

QEvent::gatherComponent
   guided by SEventConfig::GatherComp


Future
-------

Static collection of methods and values approach is convenient and
fits with the envvar controls while things are simple. 
But getting a bit awkward eg for automating max photon config 
based on available VRAM. Perhaps singleton future.


Settings
------------

EventMode
    configures what will be persisted, ie what is in the SEvt
RunningMode
    configures how running is done, eg Default/DefaultSaveG4State/RerunG4State 

MaxPhoton

MaxSimtrace

FORMER:MaxCurandState
   std::max of MaxPhoton and MaxSimtrace

MaxCurand
   configured separately, corresponds to maximum curandState slots 
   which is constrained by available VRAM 

   * using slices of curandState to give reproducible results despite 
     split launches MaxCurand not the thing that is VRAM constrained 
     instead it is constrained by the available chunked files

MaxSlot
    maximum launch slots which is constrained by available VRAM 

 
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


EventName [OPTICKS_EVENT_NAME envvar] 
    When OPTICKS_EVENT_NAME is defined it is constrained to match the build settings 
    and it also controls the default event reldir used by SEvt::save
    (requires kEventName to match the token used as part of _EventReldirDefault) 

**/


struct SYSRAP_API SEventConfig
{
    static const plog::Severity LEVEL ;  
    static constexpr const int MISSING_INDEX = std::numeric_limits<int>::max() ; 

    static const int LIMIT ; 
    static void LIMIT_Check(); 
    static std::string Desc(); 
    static std::string HitMaskLabel(); 


    // [TODO : RECONSIDER OUTDIR OUTNAME MECHANICS FOLLOWING SEVT LAYOUT 
    static const char* OutDir( const char* reldir); 
    static const char* OutDir(); 
    static const char* OutPath( const char* stem, int index, const char* ext, bool unique ); 
    static const char* OutPath( const char* reldir, const char* stem, int index, const char* ext, bool unique ); 
    static std::string DescOutPath(  const char* stem, int index, const char* ext, bool unique ) ; 
    // ]TODO

    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static constexpr const char* kIntegrationMode = "OPTICKS_INTEGRATION_MODE" ; 
    static constexpr const char* kEventMode       = "OPTICKS_EVENT_MODE" ; 
    static constexpr const char* kEventName       = "OPTICKS_EVENT_NAME" ;  
    static constexpr const char* kRunningMode     = "OPTICKS_RUNNING_MODE" ; 

    static constexpr const char* kStartIndex   = "OPTICKS_START_INDEX" ; 
    static constexpr const char* kNumEvent     = "OPTICKS_NUM_EVENT" ; 
    static constexpr const char* kNumPhoton    = "OPTICKS_NUM_PHOTON" ; 
    static constexpr const char* kNumGenstep   = "OPTICKS_NUM_GENSTEP" ; 
    static constexpr const char* kEventSkipahead  = "OPTICKS_EVENT_SKIPAHEAD" ; 

    static constexpr const char* kG4StateSpec  = "OPTICKS_G4STATE_SPEC" ; 
    static constexpr const char* kG4StateRerun = "OPTICKS_G4STATE_RERUN" ; 

    static constexpr const char* kMaxCurand    = "OPTICKS_MAX_CURAND" ; 
    static constexpr const char* kMaxSlot      = "OPTICKS_MAX_SLOT" ; 

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
    static constexpr const char* kEventReldir  = "OPTICKS_EVENT_RELDIR" ; 
    static constexpr const char* kHitMask      = "OPTICKS_HIT_MASK" ; 
    static constexpr const char* kRGMode       = "OPTICKS_RG_MODE" ; 

    // TODO: remove these, as looks like always get trumped by SEventConfig::Initialize_Comp
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
    static const char* EventName(); 
    static int         RunningMode(); 
    static const char* RunningModeLabel(); 

    static bool IsRunningModeDefault(); 
    static bool IsRunningModeG4StateSave(); 
    static bool IsRunningModeG4StateRerun();  

    static bool IsRunningModeTorch();  
    static bool IsRunningModeInputPhoton();  
    static bool IsRunningModeInputGenstep();  
    static bool IsRunningModeGun();  

    static int         EventSkipahead(); 
    static const char* G4StateSpec(); 
    static int         G4StateRerun(); 

    static int MaxCurand();  
    static int MaxSlot();  

    static int MaxGenstep(); 
    static int MaxPhoton(); 
    static int MaxSimtrace(); 

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
    static const char* EventReldir(); 
    static unsigned HitMask(); 

    static unsigned GatherComp(); 
    static unsigned SaveComp(); 

    static float PropagateEpsilon(); 

    static const char* _InputGenstepPath(int idx=-1); 
    static const char* InputGenstep(int idx=-1); 
    static bool InputGenstepPathExists(int idx); 


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
    static int NumGatherComp(); 

    static void SaveCompList( std::vector<unsigned>& save_comp ) ; 
    static int NumSaveComp(); 

    static constexpr const char* Default = "Default" ; 
    static constexpr const char* DebugHeavy = "DebugHeavy" ; 
    static constexpr const char* DebugLite = "DebugLite" ; 
    static constexpr const char* Nothing = "Nothing" ; 
    static constexpr const char* Minimal = "Minimal" ; 
    static constexpr const char* Hit = "Hit" ; 
    static constexpr const char* HitPhoton = "HitPhoton" ; 
    static constexpr const char* HitPhotonSeq = "HitPhotonSeq" ; 
    static constexpr const char* HitSeq = "HitSeq" ; 

    static void SetDefault(); 
    static void SetDebugHeavy(); 
    static void SetDebugLite(); 
    static void SetNothing();
    static void SetMinimal();
    static void SetHit();
    static void SetHitPhoton();
    static void SetHitPhotonSeq();
    static void SetHitSeq();

    static bool IsDefault(); 
    static bool IsDebugHeavy(); 
    static bool IsDebugLite(); 
    static bool IsNothing();
    static bool IsMinimal();
    static bool IsHit();
    static bool IsHitPhoton();
    static bool IsHitPhotonSeq();
    static bool IsHitSeq();

    static std::string DescEventMode() ; 

    static void SetIntegrationMode(int mode);   // IntegrationMode configures the integration of Opticks and Framework 
    static void SetEventMode(const char* mode);   // EventMode configures what will be persisted, ie what is in the SEvt
    static void SetEventName(const char* name);  
    static void SetRunningMode(const char* mode); // RunningMode configures how running is done, eg Default/DefaultSaveG4State/RerunG4State/Torch

    static void SetStartIndex(int index0); 
    static void SetNumEvent(int nevt);            // NumEvent is used by some tests 
    static void SetNumPhoton(const char* spec);   // NumPhoton is used by some tests 

    static void SetEventSkipahead(int offset);   
    static void SetG4StateSpec(const char* spec); 
    static void SetG4StateRerun(int id); 

    static void SetMaxCurand( int max_curand); 
    static void SetMaxSlot(   int max_slot); 

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
    static void SetEventReldir( const char* evt_reldir); 
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


    // STATIC VALUES SET EARLY, MANY BASED ON ENVVARS

    static int         _IntegrationModeDefault ; 
    static const char* _EventModeDefault ; 
    static const char* _EventNameDefault ; 
    static const char* _RunningModeDefault ; 
    static int         _StartIndexDefault ; 
    static int         _NumEventDefault ; 
    static const char* _NumPhotonDefault ; 
    static const char* _NumGenstepDefault ; 

    static int         _EventSkipaheadDefault ; 
    static const char* _G4StateSpecDefault ; 
    static const char* _G4StateSpecNotes ; 
    static int         _G4StateRerunDefault ; 
    static const char* _MaxBounceNotes ; 

    static const char* _MaxCurandDefault ; 
    static const char* _MaxSlotDefault ; 

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
    static const char* _EventReldirDefault ; 
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
    static const char* _EventName ; 
    static int         _RunningMode ; 
    static int         _StartIndex ; 
    static int         _NumEvent ; 

    static std::vector<int>* _GetNumPhotonPerEvent(); 
    static std::vector<int>* _NumPhotonPerEvent ; 

    static std::vector<int>* _GetNumGenstepPerEvent(); 
    static std::vector<int>* _NumGenstepPerEvent ; 

    static int               _GetNumPhoton(int idx); 
    static int               _GetNumGenstep(int idx); 


    static int               _GetNumEvent(); 
    static int               NumPhoton(int idx);  // some tests need varying photon count for each event
    static int               NumGenstep(int idx); // some tests need varying numbers of genstep for each event
    static int               NumEvent();          // some tests use event count and need to detect last event 
    static int               EventIndex(int idx) ; 
    static int               EventIndexArg(int index) ; 
    static bool              IsFirstEvent(int idx);   // 0-based idx (such as Geant4 eventID)
    static bool              IsLastEvent(int idx);    // 0-based idx (such as Geant4 eventID)

    static int         _EventSkipahead ; 
    static const char* _G4StateSpec ; 
    static int         _G4StateRerun ; 

    static int _MaxCurand ; 
    static int _MaxSlot ; 

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
    static const char* _EventReldir ; 
    static unsigned _HitMask ; 
    static int _RGMode ; 

    static unsigned _GatherComp ; 
    static unsigned _SaveComp ; 

    static float _PropagateEpsilon ;
    static const char* _InputGenstep ; 
    static const char* _InputPhoton ; 
    static const char* _InputPhotonFrame ; 



    static scontext* CONTEXT ; 
    static salloc*   ALLOC ; 
    static std::string GetGPUMeta();

    static int   Initialize_COUNT ; 
    static int   Initialize(); 
    static void  Initialize_Meta(); 
    static void  Initialize_EventName(); 
    static void  Initialize_Max(); 
    static void  Initialize_Comp(); 
    static void  Initialize_Comp_(unsigned& gather_mask, unsigned& save_mask ); 


    static constexpr const char* NAME = "SEventConfig.npy" ; 
    static NP* Serialize(); 
    static void Save(const char* dir) ; 

    static void SetDevice( size_t totalGlobalMem_bytes, std::string name ); 
    static size_t HeuristicMaxSlot(         size_t totalGlobalMem_bytes ); 
    static size_t HeuristicMaxSlot_Rounded( size_t totalGlobalMem_bytes ); 
    static std::string DescDevice(size_t totalGlobalMem_bytes, std::string name ); 

    static uint64_t EstimateAlloc(); 

}; 
 
