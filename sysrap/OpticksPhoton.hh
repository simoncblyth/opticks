#pragma once

#include <vector>
#include <map>
#include <string>

#ifdef WITH_PLOG
#include "plog/Severity.h"
#endif

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API OpticksPhoton
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
   static const char* Flag(const unsigned flag);
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




};



