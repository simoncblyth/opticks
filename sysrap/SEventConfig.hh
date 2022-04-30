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
    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static int MaxGenstep(); 
    static int MaxPhoton(); 
    static int MaxBounce(); 
    static int MaxRecord();  // full photon step record  
    static int MaxRec();     // compressed photon step record
    static int MaxSeq();     // seqhis slots
    static float MaxExtent() ; 
    static float MaxTime() ; 

    static void SetMaxGenstep(int max_genstep); 
    static void SetMaxPhoton( int max_photon); 
    static void SetMaxBounce( int max_bounce); 
    static void SetMaxRecord( int max_record); 
    static void SetMaxRec(    int max_rec); 
    static void SetMaxSeq(    int max_seq); 
    static void SetMaxExtent( float max_extent); 
    static void SetMaxTime(   float max_time ); 

    static unsigned HitMask(); 
    static void SetHitMask(const char* abrseq, char delim=',' ); 
    static std::string HitMaskDesc(); 

    static void Check(); 

    static void SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_, int max_rec_, int max_seq_ ); 
    static std::string Desc(); 

    static int _MaxGenstep ; 
    static int _MaxPhoton ; 
    static int _MaxBounce ; 
    static int _MaxRecord ; 
    static int _MaxRec ; 
    static int _MaxSeq ; 

    static float _MaxExtent ; 
    static float _MaxTime  ; 

    static unsigned _HitMask ; 
}; 




 
