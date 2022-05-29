#pragma once
#include <string>
#include "SYSRAP_API_EXPORT.hh"

/**
SEventConfig
==============

Primary user of this config is QEvent::init 


MaxPhoton
   TODO: the actual maximum depends on rngstate : need to hook into that 

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


**/


struct SYSRAP_API SEventConfig
{
    static void Check(); 
    static void SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_, int max_rec_, int max_seq_ ); 
    static std::string Desc(); 
    static std::string HitMaskLabel(); 

    static const char* OutDir( const char* reldir); 
    static const char* OutPath( const char* reldir, const char* stem, int index, const char* ext ); 

    static const char* OutDir(); 
    static const char* OutPath( const char* stem, int index, const char* ext ); 


    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static constexpr const char* kMaxGenstep = "OPTICKS_MAX_GENSTEP" ; 
    static constexpr const char* kMaxPhoton  = "OPTICKS_MAX_PHOTON" ; 
    static constexpr const char* kMaxSimtrace  = "OPTICKS_MAX_SIMTRACE" ; 
    static constexpr const char* kMaxBounce  = "OPTICKS_MAX_BOUNCE" ; 
    static constexpr const char* kMaxRecord  = "OPTICKS_MAX_RECORD" ; 
    static constexpr const char* kMaxRec     = "OPTICKS_MAX_REC" ; 
    static constexpr const char* kMaxSeq     = "OPTICKS_MAX_SEQ" ; 
    static constexpr const char* kMaxExtent  = "OPTICKS_MAX_EXTENT" ; 
    static constexpr const char* kMaxTime    = "OPTICKS_MAX_TIME" ; 
    static constexpr const char* kOutFold    = "OPTICKS_OUT_FOLD" ; 
    static constexpr const char* kOutName    = "OPTICKS_OUT_NAME" ; 
    static constexpr const char* kHitMask    = "OPTICKS_HIT_MASK" ; 
    static constexpr const char* kRGMode     = "OPTICKS_RG_MODE" ; 

    static int MaxGenstep(); 
    static int MaxPhoton(); 
    static int MaxSimtrace(); 
    static int MaxBounce(); 
    static int MaxRecord();  // full photon step record  
    static int MaxRec();     // compressed photon step record
    static int MaxSeq();     // seqhis slots
    static float MaxExtent() ; 
    static float MaxTime() ; 
    static const char* OutFold(); 
    static const char* OutName(); 
    static unsigned HitMask(); 
    static int RGMode(); 

    static bool IsRGModeRender(); 
    static bool IsRGModeSimtrace(); 
    static bool IsRGModeSimulate(); 

    static const char* RGModeLabel(); 

    static void SetMaxGenstep(int max_genstep); 
    static void SetMaxPhoton( int max_photon); 
    static void SetMaxSimtrace( int max_simtrace); 
    static void SetMaxBounce( int max_bounce); 
    static void SetMaxRecord( int max_record); 
    static void SetMaxRec(    int max_rec); 
    static void SetMaxSeq(    int max_seq); 
    static void SetMaxExtent( float max_extent); 
    static void SetMaxTime(   float max_time ); 
    static void SetOutFold( const char* out_fold); 
    static void SetOutName( const char* out_name); 
    static void SetHitMask(const char* abrseq, char delim=',' ); 
    static void SetRGMode( const char* rg_mode) ; 

    static int _MaxPhotonDefault ; 
    static int _MaxSimtraceDefault ; 

    static int _MaxGenstep ; 
    static int _MaxPhoton ; 
    static int _MaxSimtrace ; 
    static int _MaxBounce ; 
    static int _MaxRecord ; 
    static int _MaxRec ; 
    static int _MaxSeq ; 
    static float _MaxExtent ; 
    static float _MaxTime  ; 
    static const char* _OutFold ; 
    static const char* _OutName ; 
    static unsigned _HitMask ; 
    static int _RGMode ; 

}; 




 
