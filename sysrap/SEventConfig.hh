#pragma once
#include <string>
#include "SYSRAP_API_EXPORT.hh"

/**
SEventConfig
==============

MaxRecord 

   * normally 0, disabling creation of the QEvent record buffer
   * when greater than zero causes QEvent::init to create the large 
     record buffer, full recording requires a value of one greater 
     than the MaxBounce configured  (typically up to 16) 

**/


struct SYSRAP_API SEventConfig
{
    static constexpr const int M = 1000000 ; 
    static constexpr const int K = 1000 ; 

    static int MaxGenstep(); 
    static int MaxPhoton(); 
    static int MaxBounce(); 
    static int MaxRecord(); 


    static void SetMaxGenstep(int max_genstep); 
    static void SetMaxPhoton( int max_photon); 
    static void SetMaxBounce( int max_bounce); 
    static void SetMaxRecord( int max_record); 

    static unsigned HitMask(); 
    static void SetHitMask(const char* abrseq, char delim=',' ); 
    static std::string HitMaskDesc(); 

    static void Check(); 

    static void SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_ ); 
    static std::string Desc(); 

    static int _MaxGenstep ; 
    static int _MaxPhoton ; 
    static int _MaxBounce ; 
    static int _MaxRecord ; 

    static unsigned _HitMask ; 
}; 




 
