#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"

#include "SBit.hh"
#include "SStr.hh"

#ifdef WITH_PLOG
#include "PLOG.hh"
const plog::Severity OpticksPhoton::LEVEL = PLOG::EnvLevel("OpticksPhoton", "DEBUG"); 
#endif
const char* OpticksPhoton::flag2color = R"LITERAL(
    {
        "CERENKOV":"white",
        "SCINTILLATION":"white",
        "TORCH":"white",
        "MISS":"grey",
        "BULK_ABSORB":"red",
        "BULK_REEMIT":"green", 
        "BULK_SCATTER":"blue",    
        "SURFACE_DETECT":"purple",
        "SURFACE_ABSORB":"orange",      
        "SURFACE_DREFLECT":"pink",
        "SURFACE_SREFLECT":"magenta",
        "BOUNDARY_REFLECT":"yellow",
        "BOUNDARY_TRANSMIT":"cyan",
        "NAN_ABORT":"grey",
        "EFFICIENCY_COLLECT":"pink",
        "EFFICIENCY_CULL":"red"
    }
)LITERAL";






unsigned OpticksPhoton::EnumFlag(unsigned bitpos)
{
    return bitpos == 0 ? 0 : 0x1 << (bitpos - 1) ;
}

unsigned OpticksPhoton::BitPos(unsigned flag)
{
    return SBit::ffs(flag)  ;  
}

/**
OpticksPhoton::AbbrevToFlag
----------------------------

Returns lowest flag which has an abbreviation matching the argument or zero if not found. 

**/
unsigned OpticksPhoton::AbbrevToFlag( const char* abbrev )
{
    unsigned flag = 0 ; 
    if(!abbrev) return flag ;
 
    for(unsigned f=0 ; f < 32 ; f++) 
    {
        flag = EnumFlag(32-1-f) ; // <-- reverse order so unfound -> 0 
        if(strcmp(Abbrev(flag), abbrev) == 0) break ; 
    }
    return flag ;        
}

/**
OpticksPhoton::AbbrevToFlagSequence
-------------------------------------

Converts seqhis string eg "TO SR SA" into bigint 0x8ad

**/

unsigned long long OpticksPhoton::AbbrevToFlagSequence( const char* abbseq, char delim)
{
   std::vector<std::string> elem ; 
   SStr::Split(abbseq,  delim, elem ); 

   unsigned long long seqhis = 0 ; 
   for(unsigned i=0 ; i < elem.size() ; i++)
   {
       unsigned flag = AbbrevToFlag( elem[i].c_str() );
       unsigned bitpos = BitPos(flag) ; 
       unsigned long long shift = i*4 ;  
       seqhis |= ( bitpos << shift )  ; 
   }   
   return seqhis ; 
}


unsigned OpticksPhoton::AbbrevSequenceToMask( const char* abbseq, char delim)  // static
{
   std::vector<std::string> elem ; 
   SStr::Split(abbseq,  delim, elem ); 
   unsigned mask = 0 ; 

   for(unsigned i=0 ; i < elem.size() ; i++)
   {
       unsigned flag = AbbrevToFlag( elem[i].c_str() );
       mask |= flag  ; 
   }   
   return mask ; 
}

/**

OpticksPhoton::AbbrevToFlagValSequence
--------------------------------------

Convert seqmap string into two bigints, 
map values are 1-based, zero signifies None.

=====================  =============  ===========
input seqmap             seqhis         seqval 
=====================  =============  ===========
"TO:0 SR:1 SA:0"          0x8ad          0x121
"TO:0 SC: SR:1 SA:0"      0x8a6d        0x1201
=====================  =============  ===========
 
**/

void OpticksPhoton::AbbrevToFlagValSequence( unsigned long long& seqhis, unsigned long long& seqval, const char* seqmap, char edelim)
{
   seqhis = 0ull ; 
   seqval = 0ull ; 

   std::vector<std::pair<std::string, std::string> > ekv ; 
   char kvdelim=':' ; 
   SStr::ekv_split( ekv, seqmap, edelim, kvdelim );

   for(unsigned i=0 ; i < ekv.size() ; i++ ) 
   { 
       std::string skey = ekv[i].first ; 
       std::string sval = ekv[i].second ; 

       unsigned flag = AbbrevToFlag( skey.c_str() );
       unsigned bitpos = BitPos(flag) ; 

       unsigned val1 = sval.empty() ? 0 : 1u + std::atoi( sval.c_str() ) ;
 
       unsigned long long ishift = i*4 ; 

       seqhis |= ( bitpos << ishift )  ; 
       seqval |= ( val1 << ishift )  ; 

#ifdef WITH_PLOG
       LOG(debug)
                   << "[" 
                   <<  skey
                   << "] -> [" 
                   <<  sval << "]" 
                   << ( sval.empty() ? "EMPTY" : "" )
                   << " val1 " << val1 
                    ; 
#endif
    }

}



unsigned OpticksPhoton::PointVal1( const unsigned long long& seqval , unsigned bitpos )
{
    return (seqval >> bitpos*4) & 0xF ; 
}


unsigned OpticksPhoton::PointFlag( const unsigned long long& seqhis , unsigned bitpos )
{
    unsigned long long f = (seqhis >> bitpos*4) & 0xF ; 
    unsigned flg = f == 0 ? 0 : 0x1 << (f - 1) ; 
    return flg ; 
}

const char* OpticksPhoton::PointAbbrev( const unsigned long long& seqhis , unsigned bitpos )
{
    unsigned flg = PointFlag(seqhis, bitpos );
    return Abbrev(flg);     
}


std::string OpticksPhoton::FlagSequence(const unsigned long long seqhis, bool abbrev, int highlight)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);

    unsigned hi = highlight < 0 ? 16 : highlight ; 

    for(unsigned int i=0 ; i < 16 ; i++)
    {
        unsigned long long f = (seqhis >> i*4) & 0xF ; 
        unsigned int flg = f == 0 ? 0 : 0x1 << (f - 1) ; 
        if(i == hi) ss << "[" ;  
        ss << ( abbrev ? Abbrev(flg) : Flag(flg) ) ;
        if(i == hi) ss << "]" ;  
        ss << " " ; 
    }
    return ss.str();
}

/**
OpticksPhoton::FlagMask
-----------------------

A string labelling the bits set in the mskhis is returned.

**/

std::string OpticksPhoton::FlagMask(const unsigned mskhis, bool abbrev)
{
    std::vector<const char*> labels ; 

    assert( __MACHINERY == 0x1 << 17 );
    unsigned lastBit = SBit::ffs(__MACHINERY) - 1 ;  
    assert(lastBit == 17 ); 
 
    for(unsigned n=0 ; n <= lastBit ; n++ )
    {
        unsigned flag = 0x1 << n ; 
        if(mskhis & flag) labels.push_back( abbrev ? Abbrev(flag) : Flag(flag) );
    }
    unsigned nlab = labels.size() ; 

    std::stringstream ss ;
    for(unsigned i=0 ; i < nlab ; i++ ) ss << labels[i] << ( i < nlab - 1 ? "|" : ""  ) ; 
    return ss.str();
}


