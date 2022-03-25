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




const char* OpticksPhoton::_ZERO              = "  " ;
const char* OpticksPhoton::_CERENKOV          = "CK" ;
const char* OpticksPhoton::_SCINTILLATION     = "SI" ;
const char* OpticksPhoton::_TORCH             = "TO" ; 
const char* OpticksPhoton::_MISS              = "MI" ;
const char* OpticksPhoton::_BULK_ABSORB       = "AB" ;
const char* OpticksPhoton::_BULK_REEMIT       = "RE" ;
const char* OpticksPhoton::_BULK_SCATTER      = "SC" ; 
const char* OpticksPhoton::_SURFACE_DETECT    = "SD" ;
const char* OpticksPhoton::_SURFACE_ABSORB    = "SA" ; 
const char* OpticksPhoton::_SURFACE_DREFLECT  = "DR" ; 
const char* OpticksPhoton::_SURFACE_SREFLECT  = "SR" ; 
const char* OpticksPhoton::_BOUNDARY_REFLECT  = "BR" ; 
const char* OpticksPhoton::_BOUNDARY_TRANSMIT = "BT" ; 
const char* OpticksPhoton::_NAN_ABORT         = "NA" ; 
const char* OpticksPhoton::_BAD_FLAG          = "XX" ; 
const char* OpticksPhoton::_EFFICIENCY_CULL   = "EX" ; 
const char* OpticksPhoton::_EFFICIENCY_COLLECT = "EC" ; 



void OpticksPhoton::FlagAbbrevPairs( std::vector<std::pair<const char*, const char*>>& pairs )
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


/**

OpticksPhoton::Flag
--------------------

Are in process of unconflating photon flags and genstep flags 

**/

const char* OpticksPhoton::Flag(const unsigned int flag)
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

#ifdef WITH_PLOG
        LOG(debug) << "OpticksPhoton::Flag BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec ;             
#else
        srd::cerr << "OpticksPhoton::Flag BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec << std::endl ;             
#endif

    }
    return s;
}



const char* OpticksPhoton::Abbrev(const unsigned int flag)
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
                               LOG(verbose) << "OpticksPhoton::Abbrev BAD_FLAG [" << flag << "]" << std::hex << flag << std::dec ;             
    }
    return s;
}


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


