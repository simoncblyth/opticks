#pragma once

#include <vector>
#include <map>
#include <string>

#ifdef WITH_PLOG
#include "plog/Severity.h"
#endif


#ifdef STANDALONE
struct OpticksPhoton
#else
#include "SYSRAP_API_EXPORT.hh"
struct SYSRAP_API OpticksPhoton
#endif
{

#ifdef WITH_PLOG
   static const plog::Severity LEVEL ; 
#endif

   static const char* ZERO_ ;
   static const char* CERENKOV_ ;
   static const char* SCINTILLATION_ ;
   static const char* TORCH_ ;
   static const char* MISS_ ;
   static const char* BULK_ABSORB_ ;
   static const char* BULK_REEMIT_ ;
   static const char* BULK_SCATTER_ ;
   static const char* SURFACE_DETECT_ ;
   static const char* SURFACE_ABSORB_ ;
   static const char* SURFACE_DREFLECT_ ;
   static const char* SURFACE_SREFLECT_ ;
   static const char* BOUNDARY_REFLECT_ ;
   static const char* BOUNDARY_TRANSMIT_ ;
   static const char* NAN_ABORT_ ;
   static const char* EFFICIENCY_CULL_ ;
   static const char* EFFICIENCY_COLLECT_ ;
   static const char* BAD_FLAG_ ;

   static const char* Flag(const unsigned flag);

#ifdef STANDALONE
#else
   static const char* _ZERO ;
   static const char* _CERENKOV ;
   static const char* _SCINTILLATION ;
   static const char* _TORCH ;
   static const char* _MISS ;
   static const char* _BULK_ABSORB ;
   static const char* _BULK_REEMIT ;
   static const char* _BULK_SCATTER ;
   static const char* _SURFACE_DETECT ;
   static const char* _SURFACE_ABSORB ;
   static const char* _SURFACE_DREFLECT ;
   static const char* _SURFACE_SREFLECT ;
   static const char* _BOUNDARY_REFLECT ;
   static const char* _BOUNDARY_TRANSMIT ;
   static const char* _NAN_ABORT ;
   static const char* _EFFICIENCY_COLLECT ;
   static const char* _EFFICIENCY_CULL ;
   static const char* _BAD_FLAG ;

   static void FlagAbbrevPairs( std::vector<std::pair<const char*, const char*>>& pairs ) ; 
   static const char* Abbrev(const unsigned flag);
   static const char* flag2color ; 

   static unsigned EnumFlag(unsigned bitpos);
   static unsigned BitPos(unsigned flag);
   static unsigned AbbrevToFlag( const char* abbrev );
   static unsigned long long AbbrevToFlagSequence( const char* abbseq, char delim=' ');
   static unsigned AbbrevSequenceToMask( const char* abbseq, char delim=' ');
   static void AbbrevToFlagValSequence( unsigned long long& seqhis, unsigned long long& seqval, const char* seqmap, char edelim=' ') ;

   static unsigned PointVal1( const unsigned long long& seqval , unsigned bitpos );
   static unsigned PointFlag( const unsigned long long& seqhis , unsigned bitpos );
   static const char* PointAbbrev( const unsigned long long& seqhis , unsigned bitpos );
   static std::string FlagSequence(const unsigned long long seqhis, bool abbrev=true, int highlight=-1);
   static std::string FlagMask(const unsigned mskhis, bool abbrev=true);
#endif

};




const char* OpticksPhoton::ZERO_              = "." ;
const char* OpticksPhoton::CERENKOV_          = "CERENKOV" ;
const char* OpticksPhoton::SCINTILLATION_     = "SCINTILLATION" ;
const char* OpticksPhoton::MISS_              = "MISS" ;
const char* OpticksPhoton::BULK_ABSORB_       = "BULK_ABSORB" ;
const char* OpticksPhoton::BULK_REEMIT_       = "BULK_REEMIT" ;
const char* OpticksPhoton::BULK_SCATTER_      = "BULK_SCATTER" ; 
const char* OpticksPhoton::SURFACE_DETECT_    = "SURFACE_DETECT" ;
const char* OpticksPhoton::SURFACE_ABSORB_    = "SURFACE_ABSORB" ; 
const char* OpticksPhoton::SURFACE_DREFLECT_  = "SURFACE_DREFLECT" ; 
const char* OpticksPhoton::SURFACE_SREFLECT_  = "SURFACE_SREFLECT" ; 
const char* OpticksPhoton::BOUNDARY_REFLECT_  = "BOUNDARY_REFLECT" ; 
const char* OpticksPhoton::BOUNDARY_TRANSMIT_ = "BOUNDARY_TRANSMIT" ; 
const char* OpticksPhoton::TORCH_             = "TORCH" ; 
const char* OpticksPhoton::NAN_ABORT_         = "NAN_ABORT" ; 
const char* OpticksPhoton::BAD_FLAG_          = "BAD_FLAG" ; 
const char* OpticksPhoton::EFFICIENCY_CULL_     = "EFFICIENCY_CULL" ; 
const char* OpticksPhoton::EFFICIENCY_COLLECT_  = "EFFICIENCY_COLLECT" ; 


/**
OpticksPhoton::Flag
--------------------

**/

inline const char* OpticksPhoton::Flag(const unsigned int flag)
{
    const char* s = 0 ; 
    switch(flag)
    {
        case 0:                s=ZERO_;break;
        case CERENKOV:         s=CERENKOV_;break;
        case SCINTILLATION:    s=SCINTILLATION_ ;break; 
        case MISS:             s=MISS_ ;break; 
        case BULK_ABSORB:      s=BULK_ABSORB_ ;break; 
        case BULK_REEMIT:      s=BULK_REEMIT_ ;break; 
        case BULK_SCATTER:     s=BULK_SCATTER_ ;break; 
        case SURFACE_DETECT:   s=SURFACE_DETECT_ ;break; 
        case SURFACE_ABSORB:   s=SURFACE_ABSORB_ ;break; 
        case SURFACE_DREFLECT: s=SURFACE_DREFLECT_ ;break; 
        case SURFACE_SREFLECT: s=SURFACE_SREFLECT_ ;break; 
        case BOUNDARY_REFLECT: s=BOUNDARY_REFLECT_ ;break; 
        case BOUNDARY_TRANSMIT:s=BOUNDARY_TRANSMIT_ ;break; 
        case TORCH:            s=TORCH_ ;break; 
        case NAN_ABORT:        s=NAN_ABORT_ ;break; 
        case EFFICIENCY_CULL:    s=EFFICIENCY_CULL_ ;break; 
        case EFFICIENCY_COLLECT: s=EFFICIENCY_COLLECT_ ;break; 
        default:               s=BAD_FLAG_  ;

        std::cerr << "OpticksPhoton::Flag BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec << std::endl ;             
    }
    return s;
}


