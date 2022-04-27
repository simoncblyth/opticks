#pragma once
#include <string>
#include "SYSRAP_API_EXPORT.hh"

/**
SEventConfig
==============

MaxRec
    normally 0, disabling creation of the QEvent domain compressed step record buffer

MaxRecord 
    normally 0, disabling creation of the QEvent full photon step record buffer.
    When greater than zero causes QEvent::setNumPhoton to allocate the full
    step record buffer, full recording requires a value of one greater 
    than the MaxBounce configured  (typically up to 16) 

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


    static void SetMaxGenstep(int max_genstep); 
    static void SetMaxPhoton( int max_photon); 
    static void SetMaxBounce( int max_bounce); 
    static void SetMaxRecord( int max_record); 
    static void SetMaxRec(    int max_rec); 

    static unsigned HitMask(); 
    static void SetHitMask(const char* abrseq, char delim=',' ); 
    static std::string HitMaskDesc(); 

    static void Check(); 

    static void SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_, int max_rec_ ); 
    static std::string Desc(); 

    static int _MaxGenstep ; 
    static int _MaxPhoton ; 
    static int _MaxBounce ; 
    static int _MaxRecord ; 
    static int _MaxRec ; 

    static unsigned _HitMask ; 
}; 




 
