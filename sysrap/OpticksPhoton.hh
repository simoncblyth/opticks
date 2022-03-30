#pragma once

#include <vector>
#include <map>
#include <string>
#include <iostream>

#ifdef WITH_PLOG
#include "plog/Severity.h"
#endif

#include "OpticksPhoton.h"

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
   static constexpr const char* ZERO_ = ".";
   static constexpr const char* CERENKOV_ = "CERENKOV";
   static constexpr const char* SCINTILLATION_ = "SCINTILLATION" ;
   static constexpr const char* TORCH_ = "TORCH" ;
   static constexpr const char* MISS_ = "MISS" ;
   static constexpr const char* BULK_ABSORB_ = "BULK_ABSORB" ;
   static constexpr const char* BULK_REEMIT_ = "BULK_REEMIT" ;
   static constexpr const char* BULK_SCATTER_ = "BULK_SCATTER" ;
   static constexpr const char* SURFACE_DETECT_ = "SURFACE_DETECT" ;
   static constexpr const char* SURFACE_ABSORB_ = "SURFACE_ABSORB" ;
   static constexpr const char* SURFACE_DREFLECT_ = "SURFACE_DREFLECT" ;
   static constexpr const char* SURFACE_SREFLECT_ = "SURFACE_SREFLECT" ;
   static constexpr const char* BOUNDARY_REFLECT_ = "BOUNDARY_REFLECT" ;
   static constexpr const char* BOUNDARY_TRANSMIT_ = "BOUNDARY_TRANSMIT" ;
   static constexpr const char* NAN_ABORT_ = "NAN_ABORT" ;
   static constexpr const char* EFFICIENCY_CULL_ = "EFFICIENCY_CULL" ;
   static constexpr const char* EFFICIENCY_COLLECT_ = "EFFICIENCY_COLLECT" ;
   static constexpr const char* BAD_FLAG_ = "BAD_FLAG" ;

   static constexpr const char* _ZERO              = "  "  ;
   static constexpr const char* _CERENKOV          = "CK" ;
   static constexpr const char* _SCINTILLATION     = "SI" ; 
   static constexpr const char* _TORCH             = "TO" ; 
   static constexpr const char* _MISS              = "MI" ;
   static constexpr const char* _BULK_ABSORB       = "AB" ;
   static constexpr const char* _BULK_REEMIT       = "RE" ;
   static constexpr const char* _BULK_SCATTER      = "SC" ;
   static constexpr const char* _SURFACE_DETECT    = "SD" ;
   static constexpr const char* _SURFACE_ABSORB    = "SA" ;
   static constexpr const char* _SURFACE_DREFLECT  = "DR" ;
   static constexpr const char* _SURFACE_SREFLECT  = "SR" ;
   static constexpr const char* _BOUNDARY_REFLECT  = "BR" ;
   static constexpr const char* _BOUNDARY_TRANSMIT = "BT" ;
   static constexpr const char* _NAN_ABORT         = "NA" ;
   static constexpr const char* _EFFICIENCY_COLLECT = "EC" ;
   static constexpr const char* _EFFICIENCY_CULL    = "EX" ;
   static constexpr const char* _BAD_FLAG           = "XX" ;

   static const char* Flag(  const unsigned flag);
   static const char* Abbrev(const unsigned flag);
   static void FlagAbbrevPairs( std::vector<std::pair<const char*, const char*>>& pairs ) ; 

#ifdef STANDALONE
#else
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

inline const char* OpticksPhoton::Abbrev(const unsigned int flag)
{
    const char* s = 0 ; 
    switch(flag)
    {
        case 0:                s=_ZERO;break;
        case CERENKOV:         s=_CERENKOV;break;
        case SCINTILLATION:    s=_SCINTILLATION ;break; 
        case MISS:             s=_MISS ;break; 
        case BULK_ABSORB:      s=_BULK_ABSORB ;break; 
        case BULK_REEMIT:      s=_BULK_REEMIT ;break; 
        case BULK_SCATTER:     s=_BULK_SCATTER ;break; 
        case SURFACE_DETECT:   s=_SURFACE_DETECT ;break; 
        case SURFACE_ABSORB:   s=_SURFACE_ABSORB ;break; 
        case SURFACE_DREFLECT: s=_SURFACE_DREFLECT ;break; 
        case SURFACE_SREFLECT: s=_SURFACE_SREFLECT ;break; 
        case BOUNDARY_REFLECT: s=_BOUNDARY_REFLECT ;break; 
        case BOUNDARY_TRANSMIT:s=_BOUNDARY_TRANSMIT ;break; 
        case TORCH:            s=_TORCH ;break; 
        case NAN_ABORT:        s=_NAN_ABORT ;break; 
        case EFFICIENCY_COLLECT: s=_EFFICIENCY_COLLECT ;break; 
        case EFFICIENCY_CULL:    s=_EFFICIENCY_CULL ;break; 
        default:               s=_BAD_FLAG  ;
                               std::cerr << "OpticksPhoton::Abbrev BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec << std::endl ;             
    }
    return s;
}




inline void OpticksPhoton::FlagAbbrevPairs( std::vector<std::pair<const char*, const char*>>& pairs )
{
    typedef std::pair<const char*,const char*> KV ;
    pairs.push_back(KV(CERENKOV_ , _CERENKOV));
    pairs.push_back(KV(SCINTILLATION_ , _SCINTILLATION));
    pairs.push_back(KV(TORCH_ , _TORCH));
    pairs.push_back(KV(MISS_ , _MISS)); 
    pairs.push_back(KV(BULK_ABSORB_ , _BULK_ABSORB)); 
    pairs.push_back(KV(BULK_REEMIT_ , _BULK_REEMIT)); 
    pairs.push_back(KV(BULK_SCATTER_ , _BULK_SCATTER)); 
    pairs.push_back(KV(SURFACE_DETECT_ , _SURFACE_DETECT)); 
    pairs.push_back(KV(SURFACE_ABSORB_ , _SURFACE_ABSORB)); 
    pairs.push_back(KV(SURFACE_DREFLECT_ , _SURFACE_DREFLECT)); 
    pairs.push_back(KV(SURFACE_SREFLECT_ , _SURFACE_SREFLECT)); 
    pairs.push_back(KV(BOUNDARY_REFLECT_ , _BOUNDARY_REFLECT)); 
    pairs.push_back(KV(BOUNDARY_TRANSMIT_ , _BOUNDARY_TRANSMIT)); 
    pairs.push_back(KV(NAN_ABORT_ , _NAN_ABORT)); 
    pairs.push_back(KV(EFFICIENCY_CULL_ , _EFFICIENCY_CULL)); 
    pairs.push_back(KV(EFFICIENCY_COLLECT_ , _EFFICIENCY_COLLECT)); 

    // HMM: no _BAD_FLAG abbrev ?
}



