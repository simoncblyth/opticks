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

HitMask OPTICKS_HIT_MASK a comma delimited string that determines which
    subset of photons are downloaded into the "hit" array.
    Default is SD.  Settings that could be used::

        SD : SURFACE_DETECT
        EC : EFFICIENCY_COLLECT
        EX : EFFICIENCY_CULL

MaxPhoton

MaxSimtrace

MaxCurand
   Used by QRng with XORWOW running when curandState files are needed.
   With chunked curandstate controls how many chunk files and how much
   of the final chunk to load into memory.

   With OLD_MONOLITHIC_CURANDSTATE specifies which monolithic file to load.

   In both cases the value limits the total number of photons that can be
   XORWOW simulated irrespective of multi-launching to fit within VRAM.


MaxSlot OPTICKS_MAX_SLOT
    maximum CUDA launch slots

    With OPTICKS_MAX_SLOT:0 SEventConfig::HeuristicMaxSlot_Rounded is
    used to set MaxSlot to a VRAM appropriate value.

    For non-zero OPTICKS_MAX_SLOT the specified value is used.

    For large numbers of photons the value will determine how many
    launches are done.


MaxRec
    normally 0, disabling creation of the QEvent domain compressed step record buffer

MaxRecord
    normally 0, disabling creation of the QEvent full photon step record buffer.
    When greater than zero causes QEvent::setNumPhoton to allocate the full
    step record buffer, full recording requires a value of one greater
    than the MaxBounce configured  (typically up to 16)

MaxExtentDomain (mm)
    only relevant to the domain compression of the compressed step records

MaxTimeDomain (ns)
    only relevant to the domain compression of the compressed step records

MaxTime (ns)
    truncation to simulation in addition to MaxBounce



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

    static int  RecordLimit(); // sseq::SLOTS typically 32
    static void LIMIT_Check();
    static std::string Desc();

    //static std::string DescHitMask();
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
    static constexpr const char* kMaxTime      = "OPTICKS_MAX_TIME" ;

    static constexpr const char* kMaxRecord    = "OPTICKS_MAX_RECORD" ;
    static constexpr const char* kMaxRec       = "OPTICKS_MAX_REC" ;
    static constexpr const char* kMaxAux       = "OPTICKS_MAX_AUX" ;
    static constexpr const char* kMaxSup       = "OPTICKS_MAX_SUP" ;
    static constexpr const char* kMaxSeq       = "OPTICKS_MAX_SEQ" ;
    static constexpr const char* kMaxPrd       = "OPTICKS_MAX_PRD" ;
    static constexpr const char* kMaxTag       = "OPTICKS_MAX_TAG" ;
    static constexpr const char* kMaxFlat      = "OPTICKS_MAX_FLAT" ;

    static constexpr const char* kMaxExtentDomain    = "OPTICKS_MAX_EXTENT_DOMAIN" ;
    static constexpr const char* kMaxTimeDomain      = "OPTICKS_MAX_TIME_DOMAIN" ;

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
    static constexpr const char* kPropagateEpsilon0 = "OPTICKS_PROPAGATE_EPSILON0" ;
    static constexpr const char* kPropagateEpsilon0Mask = "OPTICKS_PROPAGATE_EPSILON0_MASK" ;
    static constexpr const char* kPropagateRefineDistance = "OPTICKS_PROPAGATE_REFINE_DISTANCE" ;

    static constexpr const char* kInputGenstep     = "OPTICKS_INPUT_GENSTEP" ;
    static constexpr const char* kInputGenstepSelection  = "OPTICKS_INPUT_GENSTEP_SELECTION" ;
    static constexpr const char* kInputPhoton      = "OPTICKS_INPUT_PHOTON" ;
    static constexpr const char* kInputPhotonFrame = "OPTICKS_INPUT_PHOTON_FRAME" ;
    static constexpr const char* kInputPhotonChangeTime = "OPTICKS_INPUT_PHOTON_CHANGE_TIME" ;


    static int         IntegrationMode();
    static bool        GPU_Simulation() ; // 1 or 3
    static bool        CPU_Simulation() ; // 2 or 3

    static const char* EventMode();
    static const char* EventName();
    static const char* DeviceName();
    static bool        HasDevice();

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
    static float MaxTime();

    static int MaxRecord();  // full photon step record
    static int MaxRec();     // compressed photon step record
    static int MaxAux();     // auxiliary photon step record, used for debug
    static int MaxSup();     // supplementry photon level info
    static int MaxSeq();     // seqhis slots
    static int MaxPrd();
    static int MaxTag();
    static int MaxFlat();

    static float MaxExtentDomain() ;
    static float MaxTimeDomain() ;

    static const char* OutFold();
    static const char* OutName();
    static const char* EventReldir();
    static unsigned HitMask();

    static unsigned GatherComp();
    static unsigned SaveComp();

    static float PropagateEpsilon();
    static float PropagateEpsilon0();
    static unsigned PropagateEpsilon0Mask();
    static std::string PropagateEpsilon0MaskLabel();
    static float PropagateRefineDistance();

    static const char* _InputGenstepPath(int idx=-1);
    static const char* InputGenstep(int idx=-1);
    static const char* InputGenstepSelection(int idx=-1);
    static bool InputGenstepPathExists(int idx);


    static const char* InputPhoton();
    static const char* InputPhotonFrame();
    static float       InputPhotonChangeTime();

    static int RGMode();
    static bool IsRGModeRender();
    static bool IsRGModeSimtrace();
    static bool IsRGModeSimulate();
    static bool IsRGModeTest();

    static const char* RGModeLabel();

    static std::string DescGatherComp();
    static std::string DescSaveComp();

    static void GatherCompList( std::vector<unsigned>& gather_comp ) ;
    static int NumGatherComp();

    static void SaveCompList( std::vector<unsigned>& save_comp ) ;
    static int NumSaveComp();

    //static constexpr const char* Default = "Default" ;
    static constexpr const char* DebugHeavy = "DebugHeavy" ;
    static constexpr const char* DebugLite = "DebugLite" ;
    static constexpr const char* Nothing = "Nothing" ;
    static constexpr const char* Minimal = "Minimal" ;
    static constexpr const char* Hit = "Hit" ;
    static constexpr const char* HitPhoton = "HitPhoton" ;
    static constexpr const char* HitPhotonSeq = "HitPhotonSeq" ;
    static constexpr const char* HitSeq = "HitSeq" ;

    //static void SetDefault();
    static void SetDebugHeavy();
    static void SetDebugLite();
    static void SetNothing();
    static void SetMinimal();
    static void SetHit();
    static void SetHitPhoton();
    static void SetHitPhotonSeq();
    static void SetHitSeq();

    //static bool IsDefault();
    static bool IsDebugHeavy();
    static bool IsDebugLite();
    static bool IsNothing();
    static bool IsMinimal();
    static bool IsHit();
    static bool IsHitPhoton();
    static bool IsHitPhotonSeq();
    static bool IsHitSeq();

    static bool IsMinimalOrNothing();


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
    static void SetMaxTime(   float max_time );

    static void SetMaxRecord( int max_record);
    static void SetMaxRec(    int max_rec);
    static void SetMaxAux(    int max_aux);
    static void SetMaxSup(    int max_sup);
    static void SetMaxSeq(    int max_seq);
    static void SetMaxPrd(    int max_prd);
    static void SetMaxTag(    int max_tag);
    static void SetMaxFlat(    int max_flat);

    static void SetMaxExtentDomain( float max_extent);
    static void SetMaxTimeDomain(   float max_time );




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
    static void SetPropagateEpsilon0( float eps) ;
    static void SetPropagateEpsilon0Mask( const char* abrseq, char delim=',' ) ;
    static void SetPropagateRefineDistance( float refine_distance ) ;

    static void SetInputGenstep(const char* input_genstep);
    static void SetInputGenstepSelection(const char* input_genstep_selection);
    static void SetInputPhoton(const char* input_photon);
    static void SetInputPhotonFrame(const char* input_photon_frame);
    static void SetInputPhotonChangeTime( float t0 ) ;

    static void SetGatherComp_(unsigned mask);
    static void SetGatherComp(const char* names, char delim=',') ;
    static bool GatherRecord();

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
    static const char* _MaxTimeNotes ;

    static const char* _MaxCurandDefault ;
    static const char* _MaxSlotDefault ;

    static const char* _MaxGenstepDefault ;
    static const char* _MaxPhotonDefault ;
    static const char* _MaxSimtraceDefault ;

    static int _MaxBounceDefault ;
    static float _MaxTimeDefault  ;

    static int _MaxRecordDefault ;
    static int _MaxRecDefault ;
    static int _MaxAuxDefault ;
    static int _MaxSupDefault ;
    static int _MaxSeqDefault ;
    static int _MaxPrdDefault ;
    static int _MaxTagDefault ;
    static int _MaxFlatDefault ;

    static float _MaxExtentDomainDefault ;
    static float _MaxTimeDomainDefault  ;

    static const char* _OutFoldDefault ;
    static const char* _OutNameDefault ;
    static const char* _EventReldirDefault ;
    static const char* _HitMaskDefault ;

    static const char* _RGModeDefault ;

    static const char* _GatherCompDefault ;
    static const char* _SaveCompDefault ;

    static float       _PropagateEpsilonDefault  ;
    static float       _PropagateEpsilon0Default  ;
    static const char* _PropagateEpsilon0MaskDefault ;
    static float       _PropagateRefineDistanceDefault  ;

    static const char* _InputGenstepDefault ;
    static const char* _InputGenstepSelectionDefault ;
    static const char* _InputPhotonDefault ;
    static const char* _InputPhotonFrameDefault ;
    static float       _InputPhotonChangeTimeDefault ;

    static int         _IntegrationMode ;
    static const char* _EventMode ;
    static const char* _EventName ;
    static const char* _DeviceName ;
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
    static float _MaxTime  ;

    static int _MaxRecord ;
    static int _MaxRec ;
    static int _MaxAux ;
    static int _MaxSup ;
    static int _MaxSeq ;
    static int _MaxPrd ;
    static int _MaxTag ;
    static int _MaxFlat ;

    static float _MaxExtentDomain ;
    static float _MaxTimeDomain  ;

    static const char* _OutFold ;
    static const char* _OutName ;
    static const char* _EventReldir ;
    static unsigned _HitMask ;
    static int _RGMode ;

    static unsigned _GatherComp ;
    static unsigned _SaveComp ;

    static float _PropagateEpsilon ;
    static float _PropagateEpsilon0 ;
    static unsigned _PropagateEpsilon0Mask ;
    static float _PropagateRefineDistance ;

    static const char* _InputGenstep ;
    static const char* _InputGenstepSelection ;
    static const char* _InputPhoton ;
    static const char* _InputPhotonFrame ;
    static float       _InputPhotonChangeTime ;

    static scontext* CONTEXT ;
    static salloc*   ALLOC ;
    static std::string GetGPUMeta();

    static int   Initialize_COUNT ;
    static int   Initialize();

    static void  Initialize_Meta();
    static void  Initialize_EventName();
    static void  Initialize_Comp();
    static void  Initialize_Comp_Simulate_(unsigned& gather_mask, unsigned& save_mask );
    static void  Initialize_Comp_Simtrace_(unsigned& gather_mask, unsigned& save_mask );
    static void  Initialize_Comp_Render_(  unsigned& gather_mask, unsigned& save_mask );


    static constexpr const char* NAME = "SEventConfig.npy" ;
    static NP* Serialize();
    static void Save(const char* dir) ;

    static void SetDevice( size_t totalGlobalMem_bytes, std::string name );
    static void SetDeviceName( const char* name );

    static size_t HeuristicMaxSlot(         size_t totalGlobalMem_bytes );
    static size_t HeuristicMaxSlot_Rounded( size_t totalGlobalMem_bytes );
    static std::string DescDevice(size_t totalGlobalMem_bytes, std::string name );

    static salloc*   AllocEstimate(int _max_slot=0);
    static uint64_t  AllocEstimateTotal(int _max_slot=0);

};

